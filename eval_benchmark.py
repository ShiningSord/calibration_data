import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from open_lm.hf import *
import torch.nn as nn
import argparse


def eval_zero_shot(model_name, model, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import evaluator 
    
    task_names = task_list
    model_args = f"pretrained={model_name}"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=12,
        device='cuda',
        limit=limit,
        pretrained_model=model
    )

    return results




if __name__ == "__main__":
    
    model_name = '/ossfs/workspace/yixin.jyx/output/dclm-7b-prunezero-0.6-slimpajama-seed10'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    '''
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
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #task_list = ["mmlu_flan_n_shot_loglikelihood"]
    task_list = ["boolq", "piqa", "hellaswag","winogrande", "arc_easy","arc_challenge"]
    #task_list = ["commonsense_qa"]
    #task_list = ["squadv2"]
    #task_list = ["ifeval"]
    #task_list = ["gsm8k_cot_zeroshot"]

    num_shot = 0
    results = eval_zero_shot(model_name, model, task_list, num_shot)
    print(results['results'])
