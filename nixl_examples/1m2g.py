import os
import torch
import torch.nn as nn
import nixl

# 環境變數設定（可選，開啟 debug/profiling）
os.environ["NIXL_LOG_LEVEL"] = "debug"
os.environ["NIXL_PROFILING"] = "1"

# ✅ 初始化 Nixl（僅限單機）
nixl.init()

# === 定義 Expert 模型，部署在 GPU_1 ===
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim).cuda(1)  # Expert 在 GPU 1

    def forward(self, x):
        return self.fc(x)

# === 模擬 token ===
batch_size, dim = 32, 768
tokens = torch.randn(batch_size, dim, device="cuda:0")  # tokens on GPU_0

# === 包裝 Expert，設定 remote routing ===
expert = Expert(dim)
routed_expert = nixl.RemoteModule.from_module(expert, device=torch.device("cuda:1"))

# === Routing: GPU_0 → GPU_1 expert → return to GPU_0 ===
output = routed_expert(tokens)

# === 驗證輸出正確性 ===
print(f"Output shape: {output.shape}")  # (32, 768)
print(f"Output device: {output.device}")  # should be cuda:0

# === 結束 Nixl 系統 ===
nixl.shutdown()