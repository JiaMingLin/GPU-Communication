import torch
from nixl import TransferSession

# 1. Create token (1xD) on GPU_0
D = 16
token = torch.randn(1, D, device="cuda:0")

# 2. Create expert weights (DxH) on GPU_1
H = 32
expert_weight = torch.randn(D, H, device="cuda:1")

# 3. Create transfer session
session = TransferSession()

# 4. Send token from GPU_0 → GPU_1
session.send(token, dst="cuda:1")
recv_token = session.recv_all()[0]

# 5. Execute matmul on GPU_1
with torch.cuda.device("cuda:1"):
    result = recv_token @ expert_weight  # shape = [1, H]

# 6. Send result back to GPU_0
session.send(result, dst="cuda:0")
result_on_gpu0 = session.recv_all()[0]

# 7. Return to GPU_0 for assembly
with torch.cuda.device("cuda:0"):
    print("Final result on GPU_0:", result_on_gpu0)