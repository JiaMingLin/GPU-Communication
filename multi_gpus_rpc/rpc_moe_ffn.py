#!/usr/bin/env python3
"""
Single-node Multi-GPU demos with PyTorch RPC.

Tasks
- matmul  : parallel C = A @ B with row-splitting (baseline from earlier)
- moe_ffn : coordinator routes tokens to per-worker FFN experts; workers run FFN forward

Topology
- 1 coordinator process (rank 0)
- N worker processes (ranks 1..N), each pinned to a distinct CUDA device

Highlights
- Uses TensorPipe RPC with CUDA-IPC for GPU↔GPU tensor transfer on a single node
- Parallelism: all worker RPCs launched with rpc_async and gathered with futures
- Verification switch: --verify {none,gpu,cpu1}

Run examples
  # MoE-FFN (default task), 3 workers
  python rpc_matmul_multi_gpu.py --workers 3 --tokens 4096 --d-model 1024 --d-hidden 4096 --verify gpu

  # Matmul demo, 2 workers
  python rpc_matmul_multi_gpu.py --task matmul --workers 2 --m 4096 --k 4096 --n 4096 --verify gpu

Notes
- Ensure you have ≥ workers GPUs visible, e.g. CUDA_VISIBLE_DEVICES=0,1,2
- For multi-node in the future: change --init to a rendezvous address; set device_maps both ways.
"""

from __future__ import annotations
import argparse
import os
import time
import threading
from typing import List, Tuple

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions

# ----------------------------
# RPC helpers
# ----------------------------

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    return rpc.rpc_async(rref.owner(), _call_method, args=(method, rref) + args, kwargs=kwargs)


# ----------------------------
# Worker types
# ----------------------------

_STOP_EVENT: threading.Event | None = None


class MatMulWorker:
    def __init__(self, device_index: int):
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        self.cached_B = None  # (K, N) on GPU

    @torch.no_grad()
    def cache_B(self, B: torch.Tensor) -> bool:
        assert B.is_cuda, "B must be CUDA tensor for fast GPU→GPU transfer"
        self.cached_B = B.to(self.device, non_blocking=True)
        return True

    @torch.no_grad()
    def matmul_rows(self, A_rows: torch.Tensor) -> torch.Tensor:
        assert self.cached_B is not None, "B is not cached yet; call cache_B first"
        A_rows = A_rows.to(self.device, non_blocking=True)
        C_rows = torch.matmul(A_rows, self.cached_B)
        return C_rows.cpu()


class ExpertWorker:
    """Holds one FFN expert on a single GPU and serves RPC calls."""
    def __init__(self, device_index: int):
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    @torch.no_grad()
    def set_ffn_params(self, W1: torch.Tensor, b1: torch.Tensor,
                        W2: torch.Tensor, b2: torch.Tensor) -> bool:
        self.W1 = W1.to(self.device, non_blocking=True)
        self.b1 = b1.to(self.device, non_blocking=True)
        self.W2 = W2.to(self.device, non_blocking=True)
        self.b2 = b2.to(self.device, non_blocking=True)
        return True

    @torch.no_grad()
    def ffn_forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N_tok, d_model) on coordinator GPU -> returns CPU tensor"""
        assert self.W1 is not None, "FFN not initialized"
        X = X.to(self.device, non_blocking=True)
        H = torch.matmul(X, self.W1) + self.b1
        H = torch.nn.functional.gelu(H)
        Y = torch.matmul(H, self.W2) + self.b2
        return Y.cpu()


def _signal_stop():  # called via rpc from coordinator
    global _STOP_EVENT
    if _STOP_EVENT is not None:
        _STOP_EVENT.set()
    return True


# ----------------------------
# CPU single-thread reference for verification
# ----------------------------

_prev_torch_threads: int | None = None


def _cpu_single_thread_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute A@B on CPU with a single thread (reference)."""
    global _prev_torch_threads
    A_cpu = A.detach().to("cpu", dtype=torch.float32)
    B_cpu = B.detach().to("cpu", dtype=torch.float32)
    _prev_torch_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        C_cpu = A_cpu @ B_cpu
    finally:
        if _prev_torch_threads is not None:
            torch.set_num_threads(_prev_torch_threads)
    return C_cpu


# ----------------------------
# Coordinator logic
# ----------------------------

def coordinator_main(args, world_size: int):
    opts = TensorPipeRpcBackendOptions(num_worker_threads=args.rpc_threads, init_method=args.init)
    try:
        opts.transports = ["cuda_ipc", "shm", "uv"]
        opts.channels = ["cuda_ipc", "cuda_basic", "basic"]
    except Exception:
        pass

    # Map coordinator GPU -> worker GPU for each peer (for CUDA tensor sends)
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
        if args.task == "matmul":
            rref = rpc.remote(worker_name, MatMulWorker, args=(worker_gpu,))
        elif args.task == "moe_ffn":
            rref = rpc.remote(worker_name, ExpertWorker, args=(worker_gpu,))
        else:
            raise ValueError("Unsupported task")
        workers.append((worker_name, worker_gpu, rref))

    device0 = torch.device(f"cuda:{args.coordinator_gpu}")
    torch.cuda.set_device(device0)
    torch.manual_seed(0)

    if args.task == "matmul":
        # ---------- matmul task ----------
        M, K, N = args.m, args.k, args.n
        A = torch.randn(M, K, device=device0, dtype=torch.float32)
        B = torch.randn(K, N, device=device0, dtype=torch.float32)

        # Broadcast B to workers in parallel
        t0 = time.perf_counter()
        futs = [_remote_method(MatMulWorker.cache_B, rref, B) for _, _, rref in workers]
        for f in futs: f.wait()
        t_bcast = time.perf_counter() - t0

        # Split A rows and launch
        t1 = time.perf_counter()
        row_splits = _split_rows(M, len(workers))
        result_futs: List[Tuple[slice, rpc.Future]] = []
        for (worker_name, _, rref), rows in zip(workers, row_splits):
            i, j = rows
            fut = _remote_method(MatMulWorker.matmul_rows, rref, A[i:j, :])
            result_futs.append((slice(i, j), fut))

        C = torch.empty((M, N), device="cpu", dtype=torch.float32)
        for rows, fut in result_futs:
            C_chunk = fut.wait()
            C[rows, :] = C_chunk
        t_task = time.perf_counter() - t1

    elif args.task == "moe_ffn":
        # ---------- MoE FFN task ----------
        d_model = args.d_model
        d_hidden = args.d_hidden
        num_tok = args.tokens

        # Create distinct expert params per worker (on coordinator GPU)
        expert_params = []
        for _ in workers:
            W1 = torch.randn(d_model, d_hidden, device=device0)
            b1 = torch.randn(d_hidden, device=device0)
            W2 = torch.randn(d_hidden, d_model, device=device0)
            b2 = torch.randn(d_model, device=device0)
            expert_params.append((W1, b1, W2, b2))

        # Ship to workers in parallel
        t0 = time.perf_counter()
        futs = []
        for (_, _, rref), (W1, b1, W2, b2) in zip(workers, expert_params):
            futs.append(_remote_method(ExpertWorker.set_ffn_params, rref, W1, b1, W2, b2))
        for f in futs: f.wait()
        t_bcast = time.perf_counter() - t0

        # Tokens on coordinator
        X = torch.randn(num_tok, d_model, device=device0)

        # Routing
        if args.gate == "round_robin":
            assign = torch.arange(num_tok, device=device0) % len(workers)
        elif args.gate == "random":
            g = torch.Generator(device=device0); g.manual_seed(123)
            assign = torch.randint(0, len(workers), (num_tok,), device=device0, generator=g)
        else:
            raise ValueError("--gate must be round_robin or random")

        idx_by_w: List[torch.Tensor] = []
        for w in range(len(workers)):
            idx = torch.nonzero(assign == w, as_tuple=False).flatten()
            idx_by_w.append(idx)

        # Launch parallel forwards
        t1 = time.perf_counter()
        futs2: List[Tuple[torch.Tensor, rpc.Future]] = []
        for (worker_name, _, rref), idx in zip(workers, idx_by_w):
            if idx.numel() == 0: continue
            fut = _remote_method(ExpertWorker.ffn_forward, rref, X.index_select(0, idx))
            futs2.append((idx.cpu(), fut))

        Y = torch.empty_like(X, device="cpu")
        for idx_cpu, fut in futs2:
            Y_chunk = fut.wait()
            Y.index_copy_(0, idx_cpu, Y_chunk)
        t_task = time.perf_counter() - t1

        # Local reference on GPU for verify=gpu
        with torch.no_grad():
            Y_ref_gpu = torch.empty_like(X, device=device0)
            for w, idx in enumerate(idx_by_w):
                if idx.numel() == 0: continue
                Xw = X.index_select(0, idx)
                W1, b1, W2, b2 = expert_params[w]
                Hw = torch.matmul(Xw, W1) + b1
                Hw = torch.nn.functional.gelu(Hw)
                Yw = torch.matmul(Hw, W2) + b2
                Y_ref_gpu.index_copy_(0, idx, Yw)

        # Reuse names for verify section
        C = Y
        A = None; B = None

    else:
        raise ValueError("Unsupported task")

    # ---------- Summary & verification ----------
    print("==== Summary ====")
    if args.task == "matmul":
        print(f"Task: matmul | A: {tuple(A.shape)}, B: {tuple(B.shape)}, workers: {len(workers)}")
    else:
        print(f"Task: moe_ffn | tokens: {args.tokens}, d_model: {args.d_model}, d_hidden: {args.d_hidden}, workers: {len(workers)}")
    print(f"Broadcast/Init time: {t_bcast:.3f}s")
    print(f"Parallel task time: {t_task:.3f}s")

    if args.verify.lower() != "none":
        t2 = time.perf_counter()
        if args.task == "matmul":
            if args.verify.lower() == "gpu":
                C_ref = (A @ B).cpu()
                method = "GPU baseline (coordinator)"
            elif args.verify.lower() == "cpu1":
                C_ref = _cpu_single_thread_mm(A, B)
                method = "CPU single-thread"
            else:
                raise ValueError("--verify must be one of: none, gpu, cpu1")
        else:  # moe_ffn
            if args.verify.lower() == "gpu":
                C_ref = Y_ref_gpu.detach().to("cpu")
                method = "GPU baseline (coordinator)"
            elif args.verify.lower() == "cpu1":
                # CPU single-thread FFN reference
                X_cpu = X.detach().to("cpu")
                C_ref = torch.empty_like(X_cpu)
                params_cpu = [(W1.cpu(), b1.cpu(), W2.cpu(), b2.cpu()) for (W1,b1,W2,b2) in expert_params]
                prev = torch.get_num_threads(); torch.set_num_threads(1)
                try:
                    for w, idx in enumerate(idx_by_w):
                        if idx.numel() == 0: continue
                        idx_cpu = idx.cpu()
                        Xw = X_cpu.index_select(0, idx_cpu)
                        W1, b1, W2, b2 = params_cpu[w]
                        Hw = Xw @ W1 + b1
                        Hw = torch.nn.functional.gelu(Hw)
                        Yw = Hw @ W2 + b2
                        C_ref.index_copy_(0, idx_cpu, Yw)
                finally:
                    torch.set_num_threads(prev)
                method = "CPU single-thread"
            else:
                raise ValueError("--verify must be one of: none, gpu, cpu1")
        max_abs_err = (C - C_ref).abs().max().item()
        verify_time = time.perf_counter() - t2
        print(f"Verify [{method}]: {verify_time:.3f}s, max_abs_err={max_abs_err:.3e}")
    else:
        print("Verify: skipped (--verify=none)")

    # Shutdown workers
    for rank in range(1, world_size):
        rpc.rpc_sync(f"worker{rank}", _signal_stop, args=())

    rpc.shutdown()


# ----------------------------
# Misc helpers & process entry
# ----------------------------

def _split_rows(M: int, parts: int) -> List[Tuple[int, int]]:
    q, r = divmod(M, parts)
    splits = []
    start = 0
    for p in range(parts):
        end = start + q + (1 if p < r else 0)
        splits.append((start, end))
        start = end
    return splits


def worker_main(args, rank: int, world_size: int):
    global _STOP_EVENT
    _STOP_EVENT = threading.Event()

    opts = TensorPipeRpcBackendOptions(num_worker_threads=args.rpc_threads, init_method=args.init)
    try:
        opts.transports = ["cuda_ipc", "shm", "uv"]
        opts.channels = ["cuda_ipc", "cuda_basic", "basic"]
    except Exception:
        pass

    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=opts,
    )

    _STOP_EVENT.wait()
    rpc.shutdown()


def parse_args():
    ap = argparse.ArgumentParser(description="Single-node multi-GPU RPC: matmul or MoE-FFN")
    ap.add_argument("--workers", type=int, default=2, help="Number of workers (uses this many GPUs)")

    # matmul task args
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)

    # moe_ffn task args
    ap.add_argument("--task", type=str, default="moe_ffn", choices=["matmul", "moe_ffn"], help="Which demo to run")
    ap.add_argument("--tokens", type=int, default=8192, help="# input tokens for MoE-FFN")
    ap.add_argument("--d-model", type=int, default=2048, help="Model dimension for MoE-FFN")
    ap.add_argument("--d-hidden", type=int, default=4096, help="Hidden dimension for MoE-FFN")
    ap.add_argument("--gate", type=str, default="round_robin", choices=["round_robin", "random"], help="Routing policy")

    # common
    ap.add_argument("--coordinator-gpu", type=int, default=0, help="CUDA device for coordinator")
    ap.add_argument("--rpc-threads", type=int, default=128, help="RPC worker threads")
    ap.add_argument("--init", type=str, default="tcp://127.0.0.1:29500", help="RPC init_method URL")
    ap.add_argument("--verify", type=str, default="gpu", choices=["none", "gpu", "cpu1"],
                    help="Verification: none | gpu (baseline on coordinator) | cpu1 (single-thread CPU)")
    return ap.parse_args()


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    n_devices = torch.cuda.device_count()
    assert args.workers <= n_devices, f"Need >= {args.workers} visible GPUs (have {n_devices})"

    world_size = 1 + args.workers

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
