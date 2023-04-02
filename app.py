import gradio as gr
import os, gc, torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 1024
title = "RWKV-4-Raven-7B-v6-Eng-20230401-ctx4096"

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
model_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-raven", filename=f"{title}.pth")
model = RWKV(model=model_path, strategy='cuda fp16i8 *8 -> cuda fp16')
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
    presencePenalty = 0.1,
    countPenalty = 0.1,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here

    instruction = instruction.strip()
    input = input.strip()
    ctx = generate_prompt(instruction, input)
    
    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()

examples = [
    ["Tell me about ravens.", "", 150, 1.0, 0.5, 0.4, 0.4],
    ["Write a python function to mine 1 BTC, with details and comments.", "", 150, 1.0, 0.5, 0.2, 0.2],
    ["Write a song about ravens.", "", 150, 1.0, 0.5, 0.4, 0.4],
    ["Explain the following metaphor: Life is like cats.", "", 150, 1.0, 0.5, 0.4, 0.4],
    ["Write a story using the following information", "A man named Alex chops a tree down", 150, 1.0, 0.5, 0.4, 0.4],
    ["What are the colors of these things?", "sun, moon, apple", 150, 1.0, 0.5, 0.4, 0.4],
    ["Generate a list of adjectives that describe a person as brave.", "", 150, 1.0, 0.5, 0.4, 0.4],
    ["You have $100, and your goal is to turn that into as much money as possible with AI and Machine Learning. Please respond with detailed plan.", "", 150, 1.0, 0.5, 0.4, 0.4],
]

g = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(lines=2, label="Instruction", value="Tell me about ravens."),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(minimum=10, maximum=1000, step=10, value=150), # token_count
        gr.components.Slider(minimum=0.2, maximum=2.0, step=0.1, value=1.0), # temperature
        gr.components.Slider(minimum=0, maximum=1, step=0.05, value=0.5), # top_p
        gr.components.Slider(0.0, 1.0, step=0.1, value=0.4),  # presencePenalty
        gr.components.Slider(0.0, 1.0, step=0.1, value=0.4),  # countPenalty        
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=5,
            label="Output",
        )
    ],
    title=f"🐦Raven language model",
    description="""Raven is [RWKV 7B model](https://github.com/BlinkDL/ChatRWKV) (app from [HF](https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B) finetuned to follow instructions, examples below.

For best results

- keep you prompt short and clear 
- (don't use those wordy 'You are xxxxx' ChatGPT-style prompts because such prompt styles are not in the training data yet).

""",
    examples=examples,
    cache_examples=False,
)
g.queue(concurrency_count=3, max_size=30)
g.launch(share=False, server_name='0.0.0.0')
