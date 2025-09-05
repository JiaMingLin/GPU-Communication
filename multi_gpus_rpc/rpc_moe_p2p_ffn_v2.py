#!/usr/bin/env python3
"""
Single-node MoE Layer (P2P) with Coordinator-side Verification

Changes vs. v1:
- Tokens are initialized on EACH worker's GPU (no coordinator token shipping).
- Phase 1: true worker→worker P2P token transfer via RPC (CUDA-IPC).
- Phase 2: each worker runs its FFN locally on its final token set.
- **Verification on coordinator**: coordinator pulls back each worker's final tokens (CPU) and recomputes
  outputs using the expert params it originally generated; supports:
    --verify {none, coord_gpu, coord_cpu1}

Run
  CUDA_VISIBLE_DEVICES=0,1,2 \
  python rpc_moe_p2p_ffn_v2.py --workers 3 --tokens 4096 \
    --d-model 1024 --d-hidden 4096 --gate random --verify coord_gpu
"""

from __future__ import annotations
import argparse
import time
import threading
from typing import Dict, List, Tuple

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions

# ---------------- RPC helpers ----------------

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    return rpc.rpc_async(rref.owner(), _call_method, args=(method, rref) + args, kwargs=kwargs)


# --------------- Global stop -----------------
_STOP_EVENT: threading.Event | None = None


def _signal_stop():
    global _STOP_EVENT
    if _STOP_EVENT is not None:
        _STOP_EVENT.set()
    return True


# --------------- Expert worker ---------------
class ExpertWorker:
    """One FFN expert + P2P inbox/outbox on a single GPU."""
    def __init__(self, device_index: int, name: str):
        self.name = name
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        # Expert params
        self.W1 = None; self.b1 = None; self.W2 = None; self.b2 = None
        # Token buffers
        self.X_local = None           # tokens originated here (GPU)
        self.inbox: List[torch.Tensor] = []  # received CUDA chunks
        self._pending_sends: List[rpc.Future] = []

    # ----- Model init -----
    @torch.no_grad()
    def set_ffn_params(self, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor) -> bool:
        self.W1 = W1.to(self.device, non_blocking=True)
        self.b1 = b1.to(self.device, non_blocking=True)
        self.W2 = W2.to(self.device, non_blocking=True)
        self.b2 = b2.to(self.device, non_blocking=True)
        return True

    @torch.no_grad()
    def init_tokens(self, num_tokens: int, d_model: int, seed: int = 0) -> Tuple[int, int]:
        g = torch.Generator(device=self.device); g.manual_seed(seed)
        self.X_local = torch.randn(num_tokens, d_model, device=self.device, generator=g)
        self.inbox = []
        self._pending_sends = []
        return (num_tokens, d_model)

    # ----- P2P phase -----
    @torch.no_grad()
    def p2p_send(self, dest_worker: str, idx: torch.Tensor) -> int:
        if idx.numel() == 0:
            return 0
        idx = idx.to(self.device, non_blocking=True)
        chunk = self.X_local.index_select(0, idx)
        fut = rpc.rpc_async(dest_worker, ExpertWorker.recv_tokens, args=(chunk,))
        self._pending_sends.append(fut)
        return int(chunk.shape[0])

    @torch.no_grad()
    def p2p_wait(self) -> int:
        done = 0
        for f in self._pending_sends:
            f.wait(); done += 1
        self._pending_sends.clear()
        return done

    @staticmethod
    @torch.no_grad()
    def recv_tokens(chunk: torch.Tensor) -> int:
        inst = _WORKER_SINGLETON.get()
        assert inst is not None, "Worker instance not registered"
        chunk = chunk.to(inst.device, non_blocking=True)
        inst.inbox.append(chunk)
        return int(chunk.shape[0])

    # ----- Compute phase -----
    @torch.no_grad()
    def run_ffn(self) -> Tuple[torch.Tensor, Dict[str, int]]:
        assert self.W1 is not None and self.X_local is not None
        if self.inbox:
            X_all = torch.cat([self.X_local] + self.inbox, dim=0)
        else:
            X_all = self.X_local
        H = X_all @ self.W1 + self.b1
        H = torch.nn.functional.gelu(H)
        Y = H @ self.W2 + self.b2
        stats = {"local": int(self.X_local.shape[0]),
                 "received": int(sum(x.shape[0] for x in self.inbox)),
                 "total": int(X_all.shape[0])}
        return Y.cpu(), stats

    # ----- Export tokens for coordinator verification -----
    @torch.no_grad()
    def export_tokens_cpu(self) -> torch.Tensor:
        if self.inbox:
            X_all = torch.cat([self.X_local] + self.inbox, dim=0)
        else:
            X_all = self.X_local
        return X_all.detach().to("cpu")


# registry for static recv
class _WorkerRegistry:
    def __init__(self): self.inst: ExpertWorker | None = None
    def set(self, inst: ExpertWorker): self.inst = inst
    def get(self) -> ExpertWorker | None: return self.inst

_WORKER_SINGLETON = _WorkerRegistry()


# ---------------- Coordinator ---------------

def coordinator_main(args, world_size: int):
    opts = TensorPipeRpcBackendOptions(num_worker_threads=args.rpc_threads, init_method=args.init)
    try:
        opts.transports = ["cuda_ipc", "shm", "uv"]
        opts.channels = ["cuda_ipc", "cuda_basic", "basic"]
    except Exception:
        pass

    # Device maps (coordinator → each worker). Coordinator should ONLY set maps for itself.
    for dst_rank in range(1, world_size):
        dst_name = f"worker{dst_rank}"
        # map coordinator's local GPU -> dst worker's GPU
        opts.set_device_map(dst_name, {args.coordinator_gpu: dst_rank-1})

    rpc.init_rpc(name="coordinator", rank=0, world_size=world_size, rpc_backend_options=opts)

    # Create workers
    workers = []
    for rank in range(1, world_size):
        name = f"worker{rank}"; gpu = rank - 1
        rref = rpc.remote(name, ExpertWorker, args=(gpu, name))
        workers.append((name, gpu, rref))

    # Params on coordinator GPU, then ship
    device0 = torch.device(f"cuda:{args.coordinator_gpu}")
    torch.cuda.set_device(device0)
    torch.manual_seed(0)
    d_model, d_hidden = args.d_model, args.d_hidden

    t0 = time.perf_counter()
    futs = []
    expert_params = []  # keep copy for verification
    for _, _, rref in workers:
        W1 = torch.randn(d_model, d_hidden, device=device0)
        b1 = torch.randn(d_hidden, device=device0)
        W2 = torch.randn(d_hidden, d_model, device=device0)
        b2 = torch.randn(d_model, device=device0)
        futs.append(_remote_method(ExpertWorker.set_ffn_params, rref, W1, b1, W2, b2))
        expert_params.append((W1, b1, W2, b2))
    for f in futs: f.wait()
    t_params = time.perf_counter() - t0

    # Init tokens locally
    t1 = time.perf_counter()
    futs = []
    for i, (_, _, rref) in enumerate(workers):
        futs.append(_remote_method(ExpertWorker.init_tokens, rref, args.tokens, d_model, 1234 + i))
    for f in futs: f.wait()
    t_init = time.perf_counter() - t1

    # Routing plans per worker
    t2 = time.perf_counter()
    plans: List[Dict[str, torch.Tensor]] = []
    for i, _ in enumerate(workers):
        if args.gate == "round_robin":
            assign = torch.arange(args.tokens) % len(workers)
        elif args.gate == "random":
            g = torch.Generator(); g.manual_seed(4321 + i)
            assign = torch.randint(0, len(workers), (args.tokens,), generator=g)
        else:
            raise ValueError("gate")
        plan: Dict[str, torch.Tensor] = {}
        for w_idx, (dst_name, _, _) in enumerate(workers):
            idx = torch.nonzero(assign == w_idx, as_tuple=False).flatten().to(torch.int64)
            plan[dst_name] = idx
        plans.append(plan)
    t_plan = time.perf_counter() - t2

    # Phase 1: P2P sends
    t3 = time.perf_counter()
    send_futs = []
    for (src_name, _, rref), plan in zip(workers, plans):
        for dst_name, idx in plan.items():
            send_futs.append(_remote_method(ExpertWorker.p2p_send, rref, dst_name, idx))
    for f in send_futs: f.wait()
    wait_futs = [_remote_method(ExpertWorker.p2p_wait, rref) for _, _, rref in workers]
    for f in wait_futs: f.wait()
    t_p2p = time.perf_counter() - t3

    # Phase 2: Compute
    t4 = time.perf_counter()
    out_futs = [_remote_method(ExpertWorker.run_ffn, rref) for _, _, rref in workers]
    outs = [f.wait() for f in out_futs]  # list of (Y_cpu, stats)
    t_compute = time.perf_counter() - t4

    # Verification on coordinator (pull tokens back)
    if args.verify != "none":
        v0 = time.perf_counter()
        # gather X_all per worker to CPU on coordinator
        x_futs = [_remote_method(ExpertWorker.export_tokens_cpu, rref) for _, _, rref in workers]
        X_all_cpu_list = [f.wait() for f in x_futs]
        errs: List[Tuple[str, float]] = []
        if args.verify == "coord_gpu":
            for (name, _, _), X_cpu, (Y_cpu, _), (W1, b1, W2, b2) in zip(workers, X_all_cpu_list, outs, expert_params):
                Xg = X_cpu.to(device0)
                Y_ref = torch.nn.functional.gelu(Xg @ W1 + b1) @ W2 + b2
                err = float((Y_ref.detach().to("cpu") - Y_cpu).abs().max().item())
                errs.append((name, err))
        elif args.verify == "coord_cpu1":
            prev = torch.get_num_threads(); torch.set_num_threads(1)
            try:
                for (name, _, _), X_cpu, (Y_cpu, _), (W1, b1, W2, b2) in zip(workers, X_all_cpu_list, outs, expert_params):
                    W1c, b1c = W1.detach().to("cpu"), b1.detach().to("cpu")
                    W2c, b2c = W2.detach().to("cpu"), b2.detach().to("cpu")
                    H = X_cpu @ W1c + b1c
                    H = torch.nn.functional.gelu(H)
                    Y_ref = H @ W2c + b2c
                    err = float((Y_ref - Y_cpu).abs().max().item())
                    errs.append((name, err))
            finally:
                torch.set_num_threads(prev)
        else:
            raise ValueError("--verify must be: none, coord_gpu, coord_cpu1")
        t_verify = time.perf_counter() - v0
    else:
        errs = []
        t_verify = 0.0

    # Summary
    print("==== MoE P2P FFN (Coordinator-Verify) ====")
    print(f"workers={len(workers)} | tokens/worker={args.tokens} | d_model={d_model} | d_hidden={d_hidden}")
    print(f"params:{t_params:.3f}s | init:{t_init:.3f}s | plan:{t_plan:.3f}s | p2p:{t_p2p:.3f}s | compute:{t_compute:.3f}s | verify:{t_verify:.3f}s")
    for (name, _, _), (Y_cpu, stats) in zip(workers, outs):
        err = next((e for n,e in errs if n==name), None)
        extra = f" | max_abs_err={err:.3e}" if err is not None else ""
        print(f"{name}: local={stats['local']} recv={stats['received']} total={stats['total']} out={tuple(Y_cpu.shape)}{extra}")

    # Shutdown
    for rank in range(1, world_size):
        rpc.rpc_sync(f"worker{rank}", _signal_stop, args=())
    rpc.shutdown()


# --------------- Worker entry ---------------

def worker_main(args, rank: int, world_size: int):
    global _STOP_EVENT
    _STOP_EVENT = threading.Event()

    opts = TensorPipeRpcBackendOptions(num_worker_threads=args.rpc_threads, init_method=args.init)
    try:
        opts.transports = ["cuda_ipc", "shm", "uv"]
        opts.channels = ["cuda_ipc", "cuda_basic", "basic"]
    except Exception:
        pass

    # Pairwise device maps for CUDA-IPC among workers
    name = f"worker{rank}"
    for dst_rank in range(1, world_size):
        if dst_rank == rank: continue
        opts.set_device_map(f"worker{dst_rank}", {rank-1: dst_rank-1})
    # Optional mapping to coordinator GPU 0 (not strictly needed here)
    opts.set_device_map("coordinator", {rank-1: args.coordinator_gpu})

    rpc.init_rpc(name=name, rank=rank, world_size=world_size, rpc_backend_options=opts)

    # Create and register the instance used by static recv
    inst = ExpertWorker(rank-1, name)
    _WORKER_SINGLETON.set(inst)

    _STOP_EVENT.wait()
    rpc.shutdown()


# ----------------- Launcher -----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Single-node MoE P2P FFN over PyTorch RPC (coord-verify)")
    ap.add_argument("--workers", type=int, default=2, help="Number of workers (uses this many GPUs)")
    ap.add_argument("--tokens", type=int, default=2048, help="Tokens initialized per worker (on GPU)")
    ap.add_argument("--d-model", type=int, default=1024, help="Model dimension")
    ap.add_argument("--d-hidden", type=int, default=4096, help="Hidden dimension")
    ap.add_argument("--gate", type=str, default="random", choices=["round_robin", "random"], help="Routing policy per worker")
    ap.add_argument("--verify", type=str, default="coord_gpu", choices=["none", "coord_gpu", "coord_cpu1"], help="Verification on coordinator")
    ap.add_argument("--coordinator-gpu", type=int, default=0, help="Coordinator GPU (for param broadcast and coord_gpu verify)")
    ap.add_argument("--rpc-threads", type=int, default=128, help="RPC worker threads")
    ap.add_argument("--init", type=str, default="tcp://127.0.0.1:29560", help="RPC init_method URL")
    return ap.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    ndev = torch.cuda.device_count()
    assert args.workers <= ndev, f"Need >= {args.workers} visible GPUs (have {ndev})"
    world_size = 1 + args.workers
    mp.spawn(fn=_entry, args=(args, world_size), nprocs=world_size, join=True, daemon=False)


def _entry(rank: int, args, world_size: int):
    if rank == 0:
        coordinator_main(args, world_size)
    else:
        worker_main(args, rank, world_size)


if __name__ == "__main__":
    main()
