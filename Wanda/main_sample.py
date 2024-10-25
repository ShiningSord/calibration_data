import argparse
import os 
import time 
import numpy as np
import json 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_ppl_alpaca

from lib.gptree import GPTree

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    #model.seqlen = model.config.max_position_embeddings 
    model.seqlen = 2048
    return model

def eval_zero_shot(model_name, model, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import evaluator 
    
    task_names = task_list
    model_args = f"pretrained={model_name}"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if '13b' in model_name:
        batch_size=24
    else:
        batch_size=32
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device='cuda',
        limit=limit,
        pretrained_model=model
    )

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4", "8:16"])
    parser.add_argument('--sample_mode', type=str, help='sample mode')
    parser.add_argument('--sampled_path', type=str, help='Path to sampled data')
    parser.add_argument('--sample_weight', type=str, help='sample weight')
    parser.add_argument('--sample_score', type=str, help='sample score')
    parser.add_argument('--sample_topk', type=float, help='sample topk')
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", 
                        "ablate_wanda_iter", "search"])
    parser.add_argument('--data', type=str, default='c4', help='calibration data')
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    # gradient_path 
    parser.add_argument("--gradient_path", type=str, default=None, help="Path to save the gradient.")
    parser.add_argument("--json_tree", type=str, default="data/best_tree.json", help="Path to load the json tree.")
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--eval_ppl", action="store_true")
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
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)#, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model or "70b" in args.model or "33b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)


    start_time = time.time()
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    
    end_time = time.time()
    print("pruning time: ", end_time - start_time)


    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    if args.eval_ppl:
        ppl_test = eval_ppl_alpaca(args, model, tokenizer, device)
        print(f"wikitext perplexity {ppl_test}")
    '''
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "a+") as f:
        print("method\tactual_sparsity\tppl_test\tnum_samples", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{args.nsamples}", file=f, flush=True)
    '''
    if args.eval_zero_shot:
        task_list = ["boolq", "piqa", "hellaswag","winogrande", "arc_easy","arc_challenge","mmlu_flan_n_shot_loglikelihood"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, task_list, num_shot)
        print(results['results'])

        save_filepath = os.path.join('/ossfs/workspace/Wanda/eval_result', f"log_lm_eval_{model_name}_{args.prune_method}_{args.data}_{args.sparsity_ratio}_{args.sparsity_type}_{args.nsamples}.json")
        results_json = json.dumps(results['results'], indent=4)
        with open(save_filepath, "a+") as file:
            file.write(results_json)
        print(f"Results saved to {save_filepath}")
            

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()