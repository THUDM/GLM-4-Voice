"""
A model worker with vllm libs executes the model.

Run BF16 inference with:

python vllm_model_server.py --host localhost --model-path THUDM/glm-4-voice-9b --port 10000 --dtype bfloat16 --device cuda:0

Not Supported Int4 inference.

"""
import argparse
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer
import torch
import uvicorn

from queue import Queue
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from vllm.inputs import TokensPrompt

class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt

        # variables used in the streaming process
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class ModelWorker:
    def __init__(self, model_path, dtype="bfloat16", device='cuda'):
        self.device = device
        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            device=device,
            tensor_parallel_size=1,
            dtype=dtype,
            trust_remote_code=True,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            worker_use_ray=False,
            disable_log_requests=True,
            max_model_len=8192,
        )
        self.glm_model = AsyncLLMEngine.from_engine_args(engine_args)
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @torch.inference_mode()
    async def generate_stream(self, params):
        tokenizer, model = self.glm_tokenizer, self.glm_model

        prompt = params["prompt"]
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))

        inputs = tokenizer([prompt], return_tensors="pt")
        input_ids = inputs['input_ids'][0].tolist()

        params_dict = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": -1,
            "ignore_eos": False,
            "max_tokens": max_new_tokens,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }
        sampling_params = SamplingParams(**params_dict)

        async for output in model.generate(
                TokensPrompt(**{
                    "prompt_token_ids": input_ids,
                }),
            sampling_params=sampling_params,
            request_id=f"{time.time()}"
            ):
            yield (json.dumps({"token_id": int(output.outputs[0].token_ids[-1]), "error_code": 0}) + "\n").encode()

    async def generate_stream_gate(self, params):
        try:
            async for x in self.generate_stream(params):
                yield x
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": "Server Error",
                "error_code": 1,
            }
            yield (json.dumps(ret) + "\n").encode()


app = FastAPI()


@app.post("/generate_stream")
async def generate_stream(request: Request):
    params = await request.json()

    generator = worker.generate_stream_gate(params)
    return StreamingResponse(generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    args = parser.parse_args()

    worker = ModelWorker(args.model_path, args.dtype, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
