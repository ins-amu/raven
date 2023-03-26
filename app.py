import gradio as gr
import os, gc, torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 1024
title = "RWKV-4-Pile-7B-Instruct-test4-20230326"

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
model_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-7b", filename=f"{title}.pth")
model = RWKV(model=model_path, strategy='cuda fp16i8 *20 -> cuda fp16')
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "20B_tokenizer.json")

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Input:
{input}

# Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Response:
"""

def evaluate(
    instruction,
    input=None,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    return prompt

g = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(lines=2, label="Instruction", value="Tell me about alpacas."),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(minimum=10, maximum=250, step=10, value=200),
        gr.components.Slider(minimum=0.2, maximum=2.0, step=0.1, value=1.0),
        gr.components.Slider(minimum=0, maximum=1, step=0.05, value=0.7),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title="🐦Raven-RWKV 7B",
    description="Raven-RWKV 7B is [RWKV 7B](https://github.com/BlinkDL/ChatRWKV) finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and more.",
)
g.queue(concurrency_count=1, max_size=10)
g.launch(share=False)
