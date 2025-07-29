import time
import torch
from dynamo.nixl import TransferManager

# 假設 KV cache 是 float16 的 tensor：每個 shard 為 (1024, 64, 128)
TOKEN_PER_SHARD = 1024
BATCH = 64
DIM = 128
NUM_SHARDS = 4

# 模擬 Session ID
SESSION_ID = "session123"

# 初始化 Transfer Manager
tm = TransferManager()

# Step 1: 初始化每個 Shard 在各自的 GPU 上
shards = {}
for i in range(NUM_SHARDS):
    device = torch.device(f"cuda:{i}")
    shard_data = torch.randn((TOKEN_PER_SHARD, BATCH, DIM), dtype=torch.float16, device=device)
    shards[i] = shard_data
    print(f"[Init] Shard {i} allocated on GPU {i}")

# Step 2: 模擬將所有 Shard 聚集到 GPU 1
target_gpu = 1
aggregated = []

for src_gpu in shards:
    if src_gpu == target_gpu:
        aggregated.append(shards[src_gpu])
        continue

    shard_tensor = shards[src_gpu]

    print(f"[NIXL] Moving Shard from GPU {src_gpu} to GPU {target_gpu}...")

    # 模擬 NIXL 傳輸
    torch.cuda.synchronize()
    start = time.time()

    moved_tensor = shard_tensor.to(f"cuda:{target_gpu}")  # 真實情況由 NIXL 傳輸

    torch.cuda.synchronize()
    end = time.time()

    print(f"[Done] Shard {src_gpu} transferred in {end - start:.4f} sec")
    aggregated.append(moved_tensor)

# Step 3: 拼接所有 shard 成完整 KV cache
full_kv = torch.cat(aggregated, dim=0)
print(f"[Result] Final KV cache shape: {full_kv.shape}, device: {full_kv.device}")