#!/bin/bash


model='path/to/DCLM-7B'
data='sampled_wikipedia_selfgen_2_filter'
#sampled_path='/ossfs/workspace/datacube-nas/yixin_llm/data/wikipedia_50k.json'
sampled_path='path/to/wikipedia_dclm_2_filter.json'
sparsity_ratio=0.6
#sparsity_type='2:4'
sparsity_type='unstructured'
sample_mode='selfgen'
nsamples=128
output_file="dclm_7b_${data}_ratio${sparsity_ratio}_type${sparsity_type}_nsamples${nsamples}_results.txt"


# 清空输出文件
> $output_file

for seed in {0..4}; do
    seed_value=$((seed * 5))
    echo "Running with seed: $seed" >> $output_file
    echo "Running with calibration data: $data" >> $output_file
    CUDA_VISIBLE_DEVICES=0 python main_dclm.py \
        --model $model \
        --prune_method DSnoT \
        --seed $seed_value \
        --data $data \
        --sampled_path $sampled_path \
        --sample_mode $sample_mode \
        --initial_method wanda \
        --model_type llama \
        --nsamples $nsamples \
        --sparsity_ratio ${sparsity_ratio} \
        --sparsity_type ${sparsity_type} \
        --max_cycle_time 50 \
        --update_threshold 0.1 \
        --pow_of_var_regrowing 1 \
        --eval_benchmark >> $output_file 2>&1
    echo "Finished running with seed: $seed" >> $output_file
    echo "------------------------" >> $output_file
done