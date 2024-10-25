import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import check_sparsity, prune_DSnoT
from lib.prune_opt import check_sparsity_opt, prune_DSnoT_opt
from lib.eval import eval_ppl, eval_ppl_alpaca
from lib.save_results import save_ppl_result
import json

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
        device_map="auto"
    )

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
    parser.add_argument('--model_type', type=str, default=None, help='model type, either llama or opt')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2', choices=["wikitext2", "c4", "ptb"], help='eval ppl on dataset')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity ratio.')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4", "8:16","16:32"])
    parser.add_argument("--prune_method", type=str, choices=["wanda", "sparsegpt", "magnitude", "DSnoT", "dense"])
    parser.add_argument("--initial_method", type=str, choices=["wanda", "sparsegpt", "magnitude"])
    parser.add_argument('--data', type=str, default='c4', help='calibration data')
    parser.add_argument('--sample_mode', type=str, help='sample mode')
    parser.add_argument('--sampled_path', type=str, help='Path to sampled data')
    parser.add_argument('--sample_weight', type=str, help='sample weight')
    parser.add_argument('--sample_score', type=str, help='sample score')
    parser.add_argument('--sample_topk', type=float, help='sample topk')
    parser.add_argument('--max_cycle_time', type=int, default=50, help='Max cycle time.')
    parser.add_argument('--without_DSnoT', action="store_true", help="without DSnoT")
    parser.add_argument('--update_threshold', type=float, default=0.1, help='update threshold.')
    parser.add_argument('--pow_of_var_regrowing', type=float, default=1, help='The power of variance.')
    parser.add_argument('--pow_of_var_pruning', type=float, default=1, help='The power of variance.')
    parser.add_argument("--skip_layer", type=str, default="mlp", choices=["no_skip", "mlp", "self_attn"])
    parser.add_argument("--skip_sub_layer", type=str, default="no_skip", choices=["no_skip", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "fc1", "fc2", "out_proj"])
    parser.add_argument('--without_same_sign', type=str, default="True", choices=["True", "False"], help="without same sign")
    
    parser.add_argument('--get_time_overhead', action="store_true", help="get time overhead")
    parser.add_argument('--eval_benchmark', action="store_true", help="evaluation")
    parser.add_argument('--eval_ppl', action="store_true", help="evaluation")
    parser.add_argument("--output_results_file", default="results.txt", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling different model types
    if not args.model_type:
        if ["llama kind" for model_name in ["llama", "vicuna"] if model_name in args.model]:
            args.model_type = "llama"
        elif "opt" in args.model:
            args.model_type = "opt"
        else:
            Warning("Model type not specified from model path, please specify manually")
            exit()
    print(f"model type: {args.model_type}")
    
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    
    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.model_type == "llama":
            if args.prune_method == "DSnoT":
                prune_DSnoT(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "dense":
                pass
        elif args.model_type == "opt":
            if args.prune_method == "DSnoT":
                prune_DSnoT_opt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "dense":
                pass
    
    ################################################################
    print("*"*30)
    if args.model_type == "llama":
        sparsity_ratio = check_sparsity(model)
    elif args.model_type == "opt":
        sparsity_ratio = check_sparsity_opt(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    if args.eval_ppl:
        dataset = 'wikitext2'
        ppl = eval_ppl_alpaca(model, tokenizer, dataset, device)
        print(f"\nppl on {dataset}: {ppl}\n")
        #exit()
        save_ppl_result(args, args.output_results_file, sparsity_ratio, ppl)
    #exit()
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    if args.eval_benchmark:
        task_list = ["boolq", "piqa", "hellaswag","winogrande", "arc_easy","arc_challenge","mmlu_flan_n_shot_loglikelihood"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, task_list, num_shot)
        print(results['results'])

        save_filepath = os.path.join('/ossfs/workspace/DSnoT/eval_result', f"log_lm_eval_{model_name}_{args.data}_{args.sparsity_ratio}_{args.sparsity_type}_{args.nsamples}.json")
        results_json = json.dumps(results['results'], indent=4)
        with open(save_filepath, "a+") as file:
            file.write(results_json)
        print(f"Results saved to {save_filepath}")

if __name__ == '__main__':
    main()
