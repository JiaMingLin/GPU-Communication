# moe_forward_tilepad.py
from __future__ import annotations
from collections import defaultdict
import torch

__all__ = ["moe_forward_tilepad"]

def _round_up(x: int, tile: int) -> int:
    return ((x + tile - 1) // tile) * tile

def _tile_from_dtype(dtype: torch.dtype) -> int:
    # 依常見 Tensor Core 對齊；實務可基於 GPU/庫微調
    if dtype in (torch.float16, torch.bfloat16):
        return 8
    if dtype == torch.int8:
        return 16
    return 8  # FP32/TF32 先用 8 作為保守值

@torch.inference_mode()
def moe_forward_tilepad(
    X: torch.Tensor,             # [N, d]
    expert_idx: torch.Tensor,    # [N] long
    W1: torch.Tensor,            # [E, d, d_ff]
    W2: torch.Tensor,            # [E, d_ff, d]
    activation: str = "gelu",
    group_same_n: bool = True,   # 為與 packed 保持一致之介面；此參數對 tilepad 無實質作用
    tile: int | None = None,
) -> torch.Tensor:
    """
    Tile-Padding MoE-MLP 前向：
      - 同 expert token 連續化
      - 每個 expert 的行數 m_e = round_up(n_e, tile)（n_e=0 直接跳過）
      - 以 "相同 m" 的 experts 做 batched bmm；最後只取前 n_e 行回填
    參數：
      X: [N, d]、expert_idx: [N]、W1: [E, d, d_ff]、W2: [E, d_ff, d]
      activation: "gelu" 或 "relu"
      tile: 若為 None 則依 dtype 自動選擇（FP16/BF16→8，INT8→16）
    回傳：
      Y: [N, d]（與輸入 token 原順序對齊）
    """
    assert X.dim() == 2 and expert_idx.dim() == 1
    N, d = X.shape
    E, d_in, d_ff = W1.shape
    assert d == d_in and W2.shape == (E, d_ff, d)
    assert expert_idx.shape[0] == N and expert_idx.dtype in (torch.int64, torch.long)

    tile = _tile_from_dtype(X.dtype) if tile is None else int(tile)

    # 分組與前綴和（n_e）
    n = torch.bincount(expert_idx, minlength=E)            # [E]
    order = torch.argsort(expert_idx)                      # [N]
    off_n = torch.zeros(E + 1, device=X.device, dtype=torch.int32)
    off_n[1:] = torch.cumsum(n, dim=0)

    Xg = X[order]                                          # [N, d]
    act = torch.nn.functional.gelu if activation == "gelu" else torch.nn.functional.relu

    # 決定 m_e（n_e=0 直接跳過，避免做無效 GEMM）
    m = torch.empty_like(n)
    for e in range(E):
        ne = int(n[e].item())
        m[e] = 0 if ne == 0 else _round_up(ne, tile)

    off_m = torch.zeros(E + 1, device=X.device, dtype=torch.int32)
    off_m[1:] = torch.cumsum(m, dim=0)
    M_pad = int(off_m[-1].item())

    # 建立 padded 連續輸入
    Xpad = X.new_zeros((M_pad, d))
    for e in range(E):
        ne, me = int(n[e].item()), int(m[e].item())
        if ne == 0:
            continue
        s_n, s_m = int(off_n[e].item()), int(off_m[e].item())
        Xpad[s_m:s_m+ne] = Xg[s_n:s_n+ne]     # 後段維持 0 作為 padding

    # 依 "相同 m" 分組，做 batched bmm
    Ypad = X.new_zeros((M_pad, d))
    groups: dict[int, list[int]] = defaultdict(list)
    for e in range(E):
        me = int(m[e].item())
        if me > 0:
            groups[me].append(e)

    for mval, es in groups.items():
        G  = len(es)
        Xm  = torch.stack([Xpad[int(off_m[e].item()):int(off_m[e].item())+mval] for e in es], dim=0)  # [G, m, d]
        W1m = torch.stack([W1[e] for e in es], dim=0)                                                 # [G, d, d_ff]
        W2m = torch.stack([W2[e] for e in es], dim=0)                                                 # [G, d_ff, d]
        H   = torch.bmm(Xm, W1m)                # [G, m, d_ff]
        H   = act(H)
        Y   = torch.bmm(H, W2m)                 # [G, m, d]
        for i, e in enumerate(es):
            base = int(off_m[e].item())
            Ypad[base:base+mval] = Y[i]

    # 取回有效行並還原原順序
    Yg = X.new_empty((N, d))
    for e in range(E):
        ne = int(n[e].item())
        if ne == 0:
            continue
        s_n, s_m = int(off_n[e].item()), int(off_m[e].item())
        Yg[s_n:s_n+ne] = Ypad[s_m:s_m+ne]

    Y = X.new_empty((N, d))
    Y[order] = Yg
    return Y