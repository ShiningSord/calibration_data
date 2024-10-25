import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import copy
import json
from tqdm import tqdm
from open_lm.hf import *
from datasets import load_dataset, load_from_disk


torch.set_printoptions(threshold=torch.inf)
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
    #model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )

    #model.seqlen = 2048
    return model

@torch.no_grad()
def eval_ppl(model, tokenizer, data):
    ppls = []
    n = 0
    logits_lst = []
    for text in tqdm(data):
        input_ids = tokenizer(text['text'], return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        
        ll = loss.item()
        ppl = np.exp(ll)
        ppls.append(ppl)
    return ppls


model_path='path/to/Meta-Llama-3-8B'
model_path="path/to/DCLM-7B"
data_path='path/to/dclm_llama3_8b_5k_prefix64_temp0.8_p0.95_k50_repet1.1.json'

model = get_llm(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model.eval()
device = torch.device("cuda:0")
model=model.to(device)
data = load_dataset('json',data_files=data_path)['train']

ppl = eval_ppl(model, tokenizer, data)
print(np.max(ppl))
print(np.min(ppl))
print(np.mean(ppl))
threshold = np.percentile(ppl, 80)
print(threshold)
#exit()
filtered_data = [text for text, perplexity in zip(data, ppl) if perplexity < threshold]

with open('dclm_llama3_8b_64_temp0.8_p0.95_k50_repet1.1_filter.json', 'w') as f:
    json.dump(filtered_data, f)