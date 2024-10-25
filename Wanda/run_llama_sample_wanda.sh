#!/bin/bash

model='path/to/Meta-Llama-3-8B'
#model='path/to/llama-v2-7b'
#model='path/to/llama-v2-13b'
sampled_path='path/to/dclm_llama3_8b_64_temp0.8_p0.95_k50_repet1.1_vllm_filter.json'
data='sampled_dclm_selfgen_64_temp0.8_p0.95_repet1.1_k50_filter'
sparsity_ratio=0.6
sparsity_type='unstructured'
#sparsity_type='4:8'
sample_mode='selfgen'
nsamples=128
output_file="llama3_8b_${sample_mode}_${data}_ratio${sparsity_ratio}_type${sparsity_type}_nsamples${nsamples}_results.txt"


> $output_file

for seed in {0..4}; do
    seed_value=$((seed * 5))
    echo "Running with seed: $seed" >> $output_file
    echo "Running with calibration data: $data" >> $output_file
    CUDA_VISIBLE_DEVICES=1 python main_sample.py \
        --model $model \
        --prune_method wanda \
        --seed $seed_value \
        --data $data \
        --nsamples $nsamples \
        --sampled_path $sampled_path \
        --sample_mode $sample_mode \
        --sparsity_ratio ${sparsity_ratio} \
        --sparsity_type ${sparsity_type} \
        --eval_zero_shot >> $output_file 2>&1
    echo "Finished running with seed: $seed" >> $output_file
    echo "------------------------" >> $output_file
done