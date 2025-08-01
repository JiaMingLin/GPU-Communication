import torch
import torch.nn as nn
import nixl  # Need to import Nixl first

# Ensure using 2 GPUs
assert torch.cuda.device_count() >= 2

# Step 1: Create Expert (deployed on GPU 1)
class Expert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model).cuda(1)  # Expert placed on GPU 1

    def forward(self, x):
        return self.fc(x)

# Step 2: Enable Nixl and establish channel
torch.distributed.init_process_group("nccl", init_method="env://", rank=0, world_size=1)

nixl.init()
nixl.set_default_device(torch.device("cuda:0"))  # Nixl routing entry on GPU_0

# Simulate token embeddings: placed on GPU 0
batch_size, d_model = 32, 768
tokens = torch.randn(batch_size, d_model, device="cuda:0")

# Step 3: Use Nixl routing to Expert (GPU_1)
expert = Expert(d_model)

# Wrap expert as Nixl module
routed_expert = nixl.RemoteModule.from_module(expert, device=torch.device("cuda:1"))

# Step 4: Routing + Inference
output = routed_expert(tokens)  # GPU_0 → GPU_1 → result returns to GPU_0

print("Output shape:", output.shape)  # Confirm routing success

# Step 5: Shutdown Nixl
nixl.shutdown()