import os
import sys
import argparse
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk
import json
from tqdm import tqdm
import re
from open_lm.hf import *
import torch.nn as nn


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    
    if device == "cuda":
        model.bfloat16()
        model = model.to(device)
    
    
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    

    def tokenize_function(examples):
        output = tokenizer(examples['text'])
        return output


    def truncate(example):
        example['input_ids'] = example['input_ids'][:2]
        if 'attention_mask' in example:
            example['attention_mask'] = example['attention_mask'][:2]
        return example


    def batch_generate(
        inputs,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_new_tokens=512,
        **kwargs,
    ):
        input_ids=inputs['input_ids']
        #input_ids=torch.zeros((32, 1),dtype=input_ids.dtype)
        
        with torch.no_grad():
            generation_outputs = model.generate(
                input_ids=input_ids.to(device),
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=2048,
                use_cache=True,
                repetition_penalty=1.2,
            )
            outputs = tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)
            
        return outputs


    data = load_dataset('json',data_files=args.data_path)['train']

    tokenized_datasets = data.map(
            tokenize_function,
            batched=False,
            remove_columns=['text'],
            num_proc=48
        )
    train = tokenized_datasets.map(truncate,batched=False,num_proc=48)
    
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    batch_size = 32
    loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn)
    
    lst = []
    for batch in tqdm(loader, desc="Processing"):
        generate = batch_generate(batch)
        for i in generate:
            dict = {'text': i}
            
            lst.append(dict)
    with open('path/to/wikipedia_dclm_2.json','w') as file:
        json.dump(lst, file, ensure_ascii=False, indent=4)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')

    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--data_path', type=str, default='path/to/alpaca_data_cleaned.json', help = 'path to data')
    parser.add_argument('--num_sample', type=int, default=1)

    args = parser.parse_args()
    main(args)


