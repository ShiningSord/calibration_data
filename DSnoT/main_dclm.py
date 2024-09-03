import argparse
import os 
import numpy as np
import torch
import torch.nn as nn
from open_lm.hf import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune_dclm import check_sparsity, prune_DSnoT
from lib.eval import eval_ppl
from lib.save_results import save_ppl_result

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,
    )

    model.seqlen = 2048
    return model
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--model_type', type=str, default=None, help='model type, either llama or opt')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2', choices=["wikitext2", "c4", "ptb"], help='eval ppl on dataset')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity ratio.')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["wanda", "sparsegpt", "magnitude", "DSnoT", "dense"])
    parser.add_argument("--initial_method", type=str, choices=["wanda", "sparsegpt", "magnitude"])

    parser.add_argument('--max_cycle_time', type=int, default=50, help='Max cycle time.')
    parser.add_argument('--without_DSnoT', action="store_true", help="without DSnoT")
    parser.add_argument('--update_threshold', type=float, default=0.1, help='update threshold.')
    parser.add_argument('--pow_of_var_regrowing', type=float, default=1, help='The power of variance.')
    parser.add_argument('--pow_of_var_pruning', type=float, default=1, help='The power of variance.')
    parser.add_argument("--skip_layer", type=str, default="mlp", choices=["no_skip", "mlp", "self_attn"])
    parser.add_argument("--skip_sub_layer", type=str, default="no_skip", choices=["no_skip", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "fc1", "fc2", "out_proj"])
    parser.add_argument('--without_same_sign', type=str, default="True", choices=["True", "False"], help="without same sign")
    
    parser.add_argument('--get_time_overhead', action="store_true", help="get time overhead")
    
    parser.add_argument("--output_results_file", default="results.txt", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    
    for i in range(model.config.n_layers):
        in_proj = model.model.layers[i].attention.in_proj.weight
        q_proj, k_proj, v_proj = in_proj.chunk(3,dim=0)
        model.model.layers[i].attention.q_proj.weight = nn.Parameter(q_proj)
        model.model.layers[i].attention.k_proj.weight = nn.Parameter(k_proj)
        model.model.layers[i].attention.v_proj.weight = nn.Parameter(v_proj)
        del model.model.layers[i].attention.in_proj

        w12 = model.model.layers[i].feed_forward.w12.weight
        w1, w2 = w12.chunk(2,dim=0)
        model.model.layers[i].feed_forward.w1.weight = nn.Parameter(w1)
        model.model.layers[i].feed_forward.w2.weight = nn.Parameter(w2)
        del model.model.layers[i].feed_forward.w12
    
    model.to(device)
    model.eval()
    
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "DSnoT":
            prune_DSnoT(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "dense":
            pass
        
    
    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    
    dataset = 'wikitext2'
    ppl = eval_ppl(model, tokenizer, dataset, device)
    print(f"\nppl on {dataset}: {ppl}\n")

    save_ppl_result(args, args.output_results_file, sparsity_ratio, ppl)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
