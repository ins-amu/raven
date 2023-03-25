import gradio as gr
import os, gc, torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 1024

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
    temperature=1.0,
    top_p=0.75,
    max_new_tokens=200,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    return prompt

g = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(
            minimum=1, maximum=256, step=1, value=200, label="Max tokens"
        ),
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
