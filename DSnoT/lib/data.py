# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
import numpy as np
import random
import torch
from datasets import load_dataset, load_from_disk 

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
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    traindata = load_dataset('parquet', data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/wikitext-2-raw-v1/train-00000-of-00001.parquet')
    testdata = load_dataset('parquet',data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/wikitext-2-raw-v1/test-00000-of-00001.parquet')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['train']['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['train']['text']), return_tensors='pt')

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
    traindata = load_from_disk('/ossfs/workspace/datacube-nas/yixin_llm/data/c4_train')
    valdata = load_from_disk('/ossfs/workspace/datacube-nas/yixin_llm/data/c4_validation')

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
    
# Load and process ptb dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

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
    
def get_dclm(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    #traindata = load_from_disk('/ossfs/workspace/datacube-nas/yixin_llm/data/c4_train')
    #valdata = load_from_disk('/ossfs/workspace/datacube-nas/yixin_llm/data/c4_validation')
    dataset = load_dataset('parquet',data_files='/ossfs/workspace/yixin.jyx/data/dclm-micro/output_1.parquet')
    traindata = dataset['train'].select(range(10000))
    valdata = dataset['train'].select(range(300, 600))

    trainenc = tokenizer(' '.join(traindata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        '''
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        '''
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

def get_magpie(nsamples, seed, seqlen, tokenizer):
    dataset = load_dataset('json', data_files='/ossfs/workspace/yixin.jyx/data/magpie_llama3-8b_300k_onlyans.json')
    traindata = dataset['train']#.select(range(10000))
    valdata = dataset['train'].select(range(300, 600))

    trainenc = tokenizer(' '.join(traindata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        '''
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        '''
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



def get_regenc4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('json',data_files='/ossfs/workspace/yixin.jyx/data/c4_train_llama3_8b_instruct_256.json')['train']
    valdata = load_from_disk('/ossfs/workspace/datacube-nas/yixin_llm/data/c4_validation')
    
    trainenc = tokenizer(' '.join(traindata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        '''
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        '''
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

def get_slimpajama(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets

    url='/ossfs/workspace/yixin.jyx/scale_lowrank/slimpajama/train/'
    file_pattern = 'chunk2_{i}.jsonl.zst'
    file_list = [url + file_pattern.format(i=i) for i in range(18)]   #(0,6)

    traindata = load_dataset("json", data_files={'train':file_list})['train']
    valdata = load_from_disk('/ossfs/workspace/datacube-nas/yixin_llm/data/c4_validation')

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

def get_wikipedia(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets

    traindata = load_dataset("parquet", data_files='/ossfs/workspace/datacube-nas/yixin_llm/data/wikipedia/train-00000-of-00041.parquet')['train']
    valdata = load_from_disk('/ossfs/workspace/datacube-nas/yixin_llm/data/c4_validation')

    trainenc = tokenizer(' '.join(traindata['text']), return_tensors='pt')
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        '''
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        '''
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


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if "dclm" in name:
        return get_dclm(nsamples, seed, seqlen, tokenizer)
    if "magpie" in name:
        return get_magpie(nsamples, seed, seqlen, tokenizer)
    if 'regen' in name:
        return get_regenc4(nsamples, seed, seqlen, tokenizer)
    if 'slimpajama' in name:
        return get_slimpajama(nsamples, seed, seqlen, tokenizer)
    if 'wikipedia' in name:
        return get_wikipedia(nsamples, seed, seqlen, tokenizer)