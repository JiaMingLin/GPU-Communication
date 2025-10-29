#!/usr/bin/env python3
# moe_forward_capacitypad.py
from __future__ import annotations
import math
import torch

__all__ = ["moe_forward_capacitypad"]

@torch.inference_mode()
def moe_forward_capacitypad(
    X: torch.Tensor,             # [N, d]
    expert_idx: torch.Tensor,    # [N] long, 每個 token 指派到的 expert（此實作為 k=1）
    W1: torch.Tensor,            # [E, d, d_ff]
    W2: torch.Tensor,            # [E, d_ff, d]
    activation: str = "gelu",
    group_same_n: bool = True,   # 僅為與 packed/tile 介面一致；在 Cap-Pad 中不需使用
    **kwargs,                    # 參數以 kwargs 傳入：cap 或 capacity_factor、overflow_policy 等
) -> torch.Tensor:
    """
    Capacity Padding (Cap-Pad) 的 MoE-MLP 前向實作。
    - 給定容量上限 cap（或以 capacity_factor 推得 cap），每個 expert 的 M 維補到 cap；
      對 n_e > cap 的 token 採 overflow_policy（預設 'drop'：丟棄多出的 token）。
    - 計算兩段 GEMM 後，將各 expert 有效的前 n_keep_e 行回填到輸出 Y 的原 token 位置。

    kwargs 支援：
      - cap: int，容量上限；若提供則優先使用。
      - capacity_factor: float（預設 1.0），cap = ceil(capacity_factor * ceil(N/E))。
      - overflow_policy: str in {"drop","error"}（預設 "drop"）。
      - pad_value: float（預設 0.0），padding 行的填充值（通常為 0）。
    輸出：
      - Y: [N, d]，與輸入 token 原順序對齊；若 token 被 drop，對應輸出為 0（或 pad_value 的線性映射結果，對線性層即 0）。
    備註：
      - 本函式聚焦在 MLP 計算與容量補齊；不含 gating、通訊與 k>1 聚合。
      - 若上游已確保 n_e <= cap，則本函式不會 drop。
    """
    # ---- 基本檢查 ----
    assert X.dim() == 2 and expert_idx.dim() == 1, "X:[N,d], expert_idx:[N]"
    N, d = X.shape
    E, d_in, d_ff = W1.shape
    assert d == d_in and W2.shape == (E, d_ff, d), "權重形狀不符"
    assert expert_idx.shape[0] == N and expert_idx.dtype in (torch.long, torch.int64)

    device = X.device
    dtype  = X.dtype

    # ---- 讀取 kwargs ----
    cap = kwargs.get("cap", None)
    capacity_factor = float(kwargs.get("capacity_factor", 1.0))
    overflow_policy = kwargs.get("overflow_policy", "drop")  # "drop" | "error"
    pad_value = float(kwargs.get("pad_value", 0.0))

    if cap is None:
        # 平均每 expert 的負載估計（k=1 假設）
        avg = math.ceil(N / max(1, E))
        cap = int(math.ceil(capacity_factor * avg))
    if cap <= 0:
        raise ValueError(f"cap 必須為正整數，取得 cap={cap}")

    # ---- 分組（將同 expert 的 token 連續化）----
    # 使用排序避免逐 expert nonzero 的高開銷
    counts = torch.bincount(expert_idx, minlength=E)     # [E]
    order  = torch.argsort(expert_idx)                   # [N]
    off    = torch.zeros(E + 1, device=device, dtype=torch.int32)
    off[1:] = torch.cumsum(counts, dim=0)

    # 每 expert 的保留數 n_keep_e 與 overflow
    n = counts.to(torch.int32)                           # [E]
    n_keep = torch.clamp(n, max=cap)                     # [E]
    n_over = (n - n_keep).clamp_min(0)                   # [E]
    total_keep = int(n_keep.sum().item())
    total_over = int(n_over.sum().item())

    if total_over > 0 and overflow_policy == "error":
        raise RuntimeError(f"有 {total_over} 個指派超過 cap={cap}，且 overflow_policy='error'。")

    # ---- 建立 cap 對齊的輸入批（[E, cap, d]）並填入有效行，其餘補 pad_value ----
    # 先把 X 依 order 重排：同 expert 連續
    Xg = X[order]                                        # [N, d]
    # 目標張量（一次配置），大多數情況 pad_value=0 能保證線性層輸出為 0
    Xcap = X.new_full((E, cap, d), fill_value=pad_value) # [E, cap, d]

    # 將每個 expert 的前 n_keep_e 行複製到 Xcap[e, :n_keep_e]
    # 對於 n_e > cap 的 token，若 overflow_policy='drop'，這些 token 的輸出將為 0
    for e in range(E):
        nke = int(n_keep[e].item())
        if nke == 0:
            continue
        s = int(off[e].item())
        Xcap[e, :nke] = Xg[s:s+nke]

    # ---- 兩段 GEMM：使用 batched bmm ----
    # H = Xcap @ W1  -> [E, cap, d_ff]
    # Ycap = act(H) @ W2 -> [E, cap, d]
    act = torch.nn.functional.gelu if activation == "gelu" else torch.nn.functional.relu
    # bmm 需 [E, cap, d] x [E, d, d_ff]
    H = torch.bmm(Xcap, W1)                  # [E, cap, d_ff]
    H = act(H)
    Ycap = torch.bmm(H, W2)                  # [E, cap, d]

    # ---- 回填到輸出 Y（與原 token 順序對齊）----
    Y = X.new_zeros((N, d))                  # 預設 0（被 drop 的 token 仍為 0）
    # 我們需要把每個 expert 的前 n_keep_e 行依原順序放回
    # 被 drop 的 token（每 expert 超過 cap 的尾端）不寫入（保持 0）
    for e in range(E):
        nke = int(n_keep[e].item())
        if nke == 0:
            continue
        # 原 grouped 區段起點（在 order 上）
        s = int(off[e].item())
        # 該 expert 保留的 token 的「原始索引」
        kept_indices = order[s:s+nke]        # [nke]
        # 取 Ycap 的有效前 nke 行
        Ye = Ycap[e, :nke]                   # [nke, d]
        Y[kept_indices] = Ye

    return Y


# ---------- 小測試（可直接執行） ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, E, d, dff = 1024, 8, 512, 2048
    dev, dtype = ("cuda" if torch.cuda.is_available() else "cpu"), torch.float16 if torch.cuda.is_available() else torch.float32

    X  = torch.randn(N, d, device=dev, dtype=dtype)
    W1 = torch.randn(E, d, dff, device=dev, dtype=dtype)
    W2 = torch.randn(E, dff, d, device=dev, dtype=dtype)
    expert_idx = torch.randint(0, E, (N,), device=dev, dtype=torch.long)

    # 例1：cap 充足（不 drop）
    Y1 = moe_forward_capacitypad(X, expert_idx, W1, W2, capacity_factor=4.0)
    print("OK: cap 充足，完成前向。", Y1.shape)

    # 例2：cap 緊（可能 drop）
    Y2 = moe_forward_capacitypad(X, expert_idx, W1, W2, capacity_factor=0.5, overflow_policy="drop")
    print("OK: cap 緊（可能 drop），完成前向。", Y2.shape)