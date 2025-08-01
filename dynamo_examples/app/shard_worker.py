import os
import nixl
from pydantic import BaseModel
from dynamo.sdk import service, endpoint as dynamo_endpoint

class GenerateRequest(BaseModel):
    shard_id: int
    kv_cache: bytes
    prompt: str

class GenerateResponse(BaseModel):
    output_tokens: list[str]
    new_kv_cache: bytes

@service(
    dynamo={"enabled": True, "namespace": "demo_shard", "component": "ShardWorker"},
    resources={"gpu": 1}
)
class ShardWorker:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.shard_id)
        self.transport = nixl.Transport()

    @dynamo_endpoint()
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        kv = deserialize_kv(request.kv_cache)
        out_tokens = await run_model_shard(request.prompt,
                                           shard_id=request.shard_id,
                                           kv_cache=kv)
        await self.transport.send(f"shard-{1 - request.shard_id}", serialize_kv(kv))
        return GenerateResponse(output_tokens=out_tokens,
                                new_kv_cache=serialize_kv(kv))
