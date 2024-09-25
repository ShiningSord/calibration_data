# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("parquet", data_files={"train":'../Bolaco/wikitext-2-raw-v1/train-00000-of-00001.parquet'}, split='train')
    testdata = load_dataset("parquet", data_files={"test":'../Bolaco/wikitext-2-raw-v1/test-00000-of-00001.parquet'}, split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('json', data_files = {'train': '../Bolaco/c4/c4-train.00000-of-01024.json'}, split='train')
    valdata = load_dataset('json', data_files = {'validation': '../Bolaco/c4/c4-validation.00000-of-00008.json'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_alpaca(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('parquet', data_files = {'train': '/public/home/ljt/xy/prune_llm/No_loss_pruning/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet'}, split='train')

    trainenc = tokenizer(" ".join(traindata['output']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, _

def get_wikipedia(nsamples, seed, seqlen, tokenizer):
    
    traindata = load_dataset("parquet", data_files={"train":'/public/home/ljt/xy/prune_llm/No_loss_pruning/wikipedia//train-00000-of-00041.parquet'}, split='train')
    # traindata = traindata.select(range(500))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, _

def get_slimpajama(nsamples, seed, seqlen, tokenizer):
    
    traindata = load_dataset("parquet", data_files={"train":'/public/home/ljt/xy/prune_llm/No_loss_pruning/SlimPajama-6B/train-00000-of-00048-ab2b35705f029d94.parquet'}, split='train')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, _

def get_dclm(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('json', data_files = {'train': '/public/home/ljt/xy/prune_llm/dclm-baseline-1.0/shard_00000000_processed.jsonl'}, split='train')
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, _

def get_selfgen(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('json', data_files = {'train': '/public/home/ljt/xy/prune_llm/No_loss_pruning/generated_results_new.json'}, split='train')
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, _

def get_cosmopedia(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('parquet', data_files = {'train': '/public/home/ljt/xy/prune_llm/cosmopedia-100k/train-00000-of-00002.parquet'}, split='train')
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, _

def get_pg19(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('text', data_files = {'train': '/public/home/ljt/xy/prune_llm/No_loss_pruning/PG19/data_train_files.txt'}, split='train')
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, _

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "alpaca" in name:
        return get_alpaca(nsamples, seed, seqlen, tokenizer)
    if "dclm" in name:
        return get_dclm(nsamples, seed, seqlen, tokenizer)
    if "wikipedia" in name:
        return get_wikipedia(nsamples, seed, seqlen, tokenizer)
    if "slimpajama" in name:
        return get_slimpajama(nsamples, seed, seqlen, tokenizer)
    if "selfgen" in name:
        return get_selfgen(nsamples, seed, seqlen, tokenizer)
    if "cosmopedia" in name:
        return get_cosmopedia(nsamples, seed, seqlen, tokenizer)
    if "pg19" in name:
        return get_pg19(nsamples, seed, seqlen, tokenizer)