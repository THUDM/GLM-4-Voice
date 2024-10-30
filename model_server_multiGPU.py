import argparse
import json
from threading import Thread
from queue import Queue

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers.generation.streamers import BaseStreamer
import nest_asyncio
import os


# 使当前事件循环能够嵌套
nest_asyncio.apply()


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
    def __init__(self, model_path):
        # Use Accelerate to initialize an empty model and load it efficiently for multi-GPU usage
        with init_empty_weights():
            self.glm_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        # Infer an optimal device map for multi-GPU deployment
        device_map = infer_auto_device_map(self.glm_model, max_memory={0: "16GB", 1: "16GB"})  # 根据GPU的数量和可用显存进行调整

        # Create an offload folder to store temporarily offloaded weights
        offload_folder = "./offload"
        os.makedirs(offload_folder, exist_ok=True)

        # Load model using Accelerate, dispatch to multiple GPUs, with offloading to disk
        self.glm_model = load_checkpoint_and_dispatch(
            self.glm_model,
            model_path,
            device_map=device_map,
            offload_folder=offload_folder
        )

        # Load tokenizer
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.glm_tokenizer, self.glm_model

        prompt = params["prompt"]

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))

        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)  # 将输入移动到模型所在的设备
        streamer = TokenStreamer(skip_prompt=True)
        
        thread = Thread(target=model.generate,
                        kwargs=dict(**inputs, max_new_tokens=int(max_new_tokens),
                                    temperature=float(temperature), top_p=float(top_p),
                                    streamer=streamer))
        thread.start()
        for token_id in streamer:
            yield (json.dumps({"token_id": token_id, "error_code": 0}) + "\n").encode()

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": "Server Error",
                "error_code": 1,
            }
            yield (json.dumps(ret)+ "\n").encode()


app = FastAPI()


@app.post("/generate_stream")
async def generate_stream(request: Request):
    params = await request.json()

    generator = worker.generate_stream_gate(params)
    return StreamingResponse(generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    args = parser.parse_args()

    worker = ModelWorker(args.model_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")