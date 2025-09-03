#!/usr/bin/env python3
"""
Single-node Multi-GPU matrix multiplication with PyTorch RPC.

Topology
- 1 coordinator process (rank 0)
- N worker processes (ranks 1..N), each pinned to a distinct CUDA device

Highlights
- Uses TensorPipe RPC with CUDA-IPC for GPU↔GPU tensor transfer on a single node
- Broadcasts matrix B once to every worker (in parallel)
- Splits A by rows and launches parallel RPC calls to compute C_i = A_i @ B on workers
- Gathers the row blocks back and verifies correctness

Run
  python rpc_matmul_multi_gpu.py \
    --workers 2 --m 4096 --k 4096 --n 4096 \
    --coordinator-gpu 0 --init tcp://127.0.0.1:29501

Notes
- Ensure you have ≥ workers GPUs visible, e.g. CUDA_VISIBLE_DEVICES=0,1,2
- To emphasize parallelism, we launch all RPCs with rpc_async and wait_all.
- Results are returned on CPU to avoid configuring reverse CUDA device maps.
- For multi-node in the future, replace init_method/addresses, and set device_maps accordingly.
"""

from __future__ import annotations
import argparse
import math
import os
import time
import threading
from typing import Dict, List, Tuple

# For optional single-thread CPU reference
_prev_torch_threads: int | None = None

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions

# ----------------------------
# Utilities for RRef method calls
# ----------------------------

def _call_method(method, rref, *args, **kwargs):
    """Execute a method on the local reference of a remote object."""
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    """Helper to call a method via RPC on an RRef-owned worker."""
    return rpc.rpc_async(rref.owner(), _call_method, args=(method, rref) + args, kwargs=kwargs)


# ----------------------------
# Worker-side implementation
# ----------------------------

_STOP_EVENT: threading.Event | None = None


class MatMulWorker:
    def __init__(self, device_index: int):
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        self.cached_B = None  # (K, N) on GPU

    @torch.no_grad()
    def cache_B(self, B: torch.Tensor) -> bool:
        """Cache matrix B on the worker GPU (called once)."""
        assert B.is_cuda, "B must be CUDA tensor for fast GPU→GPU transfer"
        self.cached_B = B.to(self.device, non_blocking=True)
        return True

    @torch.no_grad()
    def matmul_rows(self, A_rows: torch.Tensor) -> torch.Tensor:
        """Compute A_rows @ B using the cached B. Returns CPU tensor for simpler gather."""
        assert self.cached_B is not None, "B is not cached yet; call cache_B first"
        # Move rows to worker GPU and multiply
        A_rows = A_rows.to(self.device, non_blocking=True)
        C_rows = torch.matmul(A_rows, self.cached_B)
        # Return to CPU to avoid needing a reverse device map
        return C_rows.cpu()


# simple RPC-exposed function to signal worker shutdown

def _signal_stop():  # called via rpc from coordinator
    global _STOP_EVENT
    if _STOP_EVENT is not None:
        _STOP_EVENT.set()
    return True


# ----------------------------
# Coordinator logic
# ----------------------------

def _cpu_single_thread_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute A@B on CPU with a single thread (reference)."""
    global _prev_torch_threads
    A_cpu = A.detach().to("cpu", dtype=torch.float32)
    B_cpu = B.detach().to("cpu", dtype=torch.float32)
    # Save & force single-thread
    _prev_torch_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        C_cpu = A_cpu @ B_cpu
    finally:
        # restore threads
        if _prev_torch_threads is not None:
            torch.set_num_threads(_prev_torch_threads)
    return C_cpu

@torch.no_grad()
def coordinator_main(args, world_size: int):
    # Build device maps so coordinator can send CUDA tensors from coordinator GPU to each worker GPU.
    opts = TensorPipeRpcBackendOptions(num_worker_threads=args.rpc_threads, init_method=args.init)
    # prefer CUDA IPC, shared memory, and TCP fallback
    try:
        opts.transports = ["cuda_ipc", "shm", "uv"]
        opts.channels = ["cuda_ipc", "cuda_basic", "basic"]
    except Exception:
        pass

    # Map coordinator GPU -> worker GPU for each peer
    for rank in range(1, world_size):
        worker_name = f"worker{rank}"
        worker_gpu = rank - 1
        opts.set_device_map(worker_name, {args.coordinator_gpu: worker_gpu})

    rpc.init_rpc(
        name="coordinator",
        rank=0,
        world_size=world_size,
        rpc_backend_options=opts,
    )

    # Create remote worker modules (one per worker)
    workers = []
    for rank in range(1, world_size):
        worker_name = f"worker{rank}"
        worker_gpu = rank - 1
        rref = rpc.remote(worker_name, MatMulWorker, args=(worker_gpu,))
        workers.append((worker_name, worker_gpu, rref))

    # Prepare matrices on coordinator GPU (to leverage CUDA→CUDA broadcast)
    device0 = torch.device(f"cuda:{args.coordinator_gpu}")
    torch.cuda.set_device(device0)

    M, K, N = args.m, args.k, args.n
    torch.manual_seed(0)
    A = torch.randn(M, K, device=device0, dtype=torch.float32)
    B = torch.randn(K, N, device=device0, dtype=torch.float32)

    # Broadcast B to workers in parallel
    t0 = time.perf_counter()
    futs = []
    for _, _, rref in workers:
        futs.append(_remote_method(MatMulWorker.cache_B, rref, B))
    for f in futs:
        f.wait()
    t_bcast = time.perf_counter() - t0

    # Split A by rows and launch parallel matmul RPCs
    t1 = time.perf_counter()
    row_splits = _split_rows(M, len(workers))

    result_futs: List[Tuple[slice, rpc.Future]] = []
    start = 0
    for (worker_name, _, rref), rows in zip(workers, row_splits):
        i, j = rows
        A_chunk = A[i:j, :]  # stays on coordinator GPU; sent CUDA→CUDA
        fut = _remote_method(MatMulWorker.matmul_rows, rref, A_chunk)
        result_futs.append((slice(i, j), fut))

    # Gather
    C = torch.empty((M, N), device="cpu", dtype=torch.float32)
    for rows, fut in result_futs:
        C_chunk = fut.wait()  # CPU tensor
        C[rows, :] = C_chunk
    t_matmul = time.perf_counter() - t1

    # Optional verification
    print("==== Summary ====")
    print(f"A: {tuple(A.shape)}, B: {tuple(B.shape)}, workers: {len(workers)}")
    print(f"Broadcast B time: {t_bcast:.3f}s")
    print(f"Parallel matmul time: {t_matmul:.3f}s")

    if args.verify.lower() != "none":
        t2 = time.perf_counter()
        if args.verify.lower() == "gpu":
            C_ref = (A @ B).cpu()
            method = "GPU baseline (coordinator)"
        elif args.verify.lower() == "cpu1":
            C_ref = _cpu_single_thread_mm(A, B)
            method = "CPU single-thread"
        else:
            raise ValueError("--verify must be one of: none, gpu, cpu1")
        max_abs_err = (C - C_ref).abs().max().item()
        verify_time = time.perf_counter() - t2
        print(f"Verify [{method}]: {verify_time:.3f}s, max_abs_err={max_abs_err:.3e}")
    else:
        print("Verify: skipped (--verify=none)")

    # Tell workers to stop and then shutdown RPC
    for rank in range(1, world_size):
        rpc.rpc_sync(f"worker{rank}", _signal_stop, args=())

    rpc.shutdown()


def _split_rows(M: int, parts: int) -> List[Tuple[int, int]]:
    q, r = divmod(M, parts)
    splits = []
    start = 0
    for p in range(parts):
        end = start + q + (1 if p < r else 0)
        splits.append((start, end))
        start = end
    return splits


# ----------------------------
# Worker process entry
# ----------------------------

def worker_main(args, rank: int, world_size: int):
    global _STOP_EVENT
    _STOP_EVENT = threading.Event()

    opts = TensorPipeRpcBackendOptions(num_worker_threads=args.rpc_threads, init_method=args.init)
    try:
        opts.transports = ["cuda_ipc", "shm", "uv"]
        opts.channels = ["cuda_ipc", "cuda_basic", "basic"]
    except Exception:
        pass

    # Workers may also send CUDA tensors to coordinator in other tasks; set reverse map if needed.
    # Not strictly required here since we return CPU results.
    # Example:
    # worker_gpu = rank - 1
    # opts.set_device_map("coordinator", {worker_gpu: args.coordinator_gpu})

    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=opts,
    )

    # Block until coordinator signals stop
    _STOP_EVENT.wait()

    rpc.shutdown()


# ----------------------------
# Launcher
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Single-node multi-GPU matmul with PyTorch RPC")
    ap.add_argument("--workers", type=int, default=2, help="Number of workers (uses this many GPUs)")
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--coordinator-gpu", type=int, default=0, help="CUDA device for coordinator")
    ap.add_argument("--rpc-threads", type=int, default=128, help="RPC worker threads")
    ap.add_argument("--init", type=str, default="tcp://127.0.0.1:29500", help="RPC init_method URL")
    ap.add_argument("--verify", type=str, default="gpu", choices=["none", "gpu", "cpu1"],
                    help="Verification method: none | gpu (A@B on coordinator GPU) | cpu1 (single-thread CPU)")
    return ap.parse_args()


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    n_devices = torch.cuda.device_count()
    assert args.workers <= n_devices, f"Need >= {args.workers} visible GPUs (have {n_devices})"

    world_size = 1 + args.workers

    # Spawn processes; rank 0 is coordinator, 1..N are workers
    mp.spawn(
        fn=_entry,
        args=(args, world_size),
        nprocs=world_size,
        join=True,
        daemon=False,
    )


def _entry(rank: int, args, world_size: int):
    if rank == 0:
        coordinator_main(args, world_size)
    else:
        worker_main(args, rank, world_size)


if __name__ == "__main__":
    main()
