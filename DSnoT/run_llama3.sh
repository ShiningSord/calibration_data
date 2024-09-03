#model='/ossfs/workspace/common_base_model/Meta-Llama-3-8B-Instruct'
model='/ossfs/workspace/datacube-nas/yixin_llm/Meta-Llama-3-8B'

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model $model \
    --prune_method DSnoT \
    --seed 10 \
    --initial_method wanda \
    --model_type llama \
    --nsamples 1024 \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --max_cycle_time 50 \
    --update_threshold 0.1 \
    --pow_of_var_regrowing 1 \
    --save_model '/mntnlp/yixin.jyx/yixin.jyxx/yixin.jyx/output/llama3-8b-DSnoT-wanda-4:8-dclm-seed10'