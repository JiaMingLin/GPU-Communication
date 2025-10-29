# moe_forward_packed.py
from __future__ import annotations
from collections import defaultdict
import torch

__all__ = ["moe_forward_packed"]

@torch.inference_mode()
def moe_forward_packed(
    X: torch.Tensor,             # [N, d]
    expert_idx: torch.Tensor,    # [N] long
    W1: torch.Tensor,            # [E, d, d_ff]
    W2: torch.Tensor,            # [E, d_ff, d]
    activation: str = "gelu",
    group_same_n: bool = True,
    tile: int | None = None,     # 保持與 tilepad 一致；此函式不使用
) -> torch.Tensor:
    """
    Packed（不補零）MoE-MLP 前向：
      - 將相同 expert 的 token 連續化
      - 對每個 expert 以實際行數 n_e 執行兩段 GEMM
      - （選用）相同 n 的 experts 以 batched bmm 合併，降低 kernel 啟動成本
    參數：
      X: [N, d]、expert_idx: [N]、W1: [E, d, d_ff]、W2: [E, d_ff, d]
      activation: "gelu" 或 "relu"
      group_same_n: 將相同 n 的 experts 用 batched bmm
    回傳：
      Y: [N, d]（與輸入 token 原順序對齊）
    """
    assert X.dim() == 2 and expert_idx.dim() == 1
    N, d = X.shape
    E, d_in, d_ff = W1.shape
    assert d == d_in and W2.shape == (E, d_ff, d)
    assert expert_idx.shape[0] == N and expert_idx.dtype in (torch.int64, torch.long)

    # 分組與前綴和
    counts = torch.bincount(expert_idx, minlength=E)            # [E]
    order  = torch.argsort(expert_idx)                          # [N]：同 expert 連續
    off    = torch.zeros(E + 1, device=X.device, dtype=torch.int32)
    off[1:] = torch.cumsum(counts, dim=0)

    Xg = X[order]                                               # [N, d]
    Yg = Xg.new_empty((N, d))
    act = torch.nn.functional.gelu if activation == "gelu" else torch.nn.functional.relu

    # 以 "相同 n" 分組
    groups: dict[int, list[int]] = defaultdict(list)
    for e in range(E):
        ne = int(counts[e].item())
        if ne > 0:
            groups[ne].append(e)

    for ne, es in groups.items():
        if (not group_same_n) or len(es) == 1:
            for e in es:
                s = int(off[e].item())
                Xe = Xg[s:s+ne]                      # [ne, d]
                He = Xe @ W1[e]                      # [ne, d_ff]
                He = act(He)
                Ye = He @ W2[e]                      # [ne, d]
                Yg[s:s+ne] = Ye
        else:
            G  = len(es)
            Xb = torch.stack([Xg[int(off[e].item()):int(off[e].item())+ne] for e in es], dim=0)  # [G, ne, d]
            W1b = torch.stack([W1[e] for e in es], dim=0)                                        # [G, d, d_ff]
            W2b = torch.stack([W2[e] for e in es], dim=0)                                        # [G, d_ff, d]
            Hb = torch.bmm(Xb, W1b)            # [G, ne, d_ff]
            Hb = act(Hb)
            Yb = torch.bmm(Hb, W2b)            # [G, ne, d]
            for i, e in enumerate(es):
                s = int(off[e].item())
                Yg[s:s+ne] = Yb[i]

    # 還原原順序
    Y = X.new_empty((N, d))
    Y[order] = Yg
    return Y