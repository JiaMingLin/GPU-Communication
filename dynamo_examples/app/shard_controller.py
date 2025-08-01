from dynamo.runtime import DistributedRuntime
import asyncio
from pydantic import BaseModel

class GenerateResponse(BaseModel):
    output_tokens: list[str]
    new_kv_cache: bytes

async def main(prompt: str):
    runtime = DistributedRuntime()

    resp0 = await runtime.call("demo_shard/ShardWorker/generate", {
        "shard_id": 0, "prompt": prompt, "kv_cache": b""})
    resp1 = await runtime.call("demo_shard/ShardWorker/generate", {
        "shard_id": 1, "prompt": prompt, "kv_cache": resp0["new_kv_cache"]})

    return resp0["output_tokens"] + resp1["output_tokens"]

if __name__ == "__main__":
    tokens = asyncio.run(main("Hello from 2‑GPU shard demo"))
    print("Generated tokens:", tokens)
