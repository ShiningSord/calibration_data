#!/bin/bash


model='path/to/DCLM-7B'
data='sampled_wikipedia_selfgen_1_filter'
sampled_path='path/to/wikipedia_dclm_1_filter.json'
sparsity_ratio=0.6
#sparsity_type='2:4'
sparsity_type='unstructured'
sample_mode='selfgen'
nsamples=128
output_file="dclm_7b_sparsegpt_${sample_mode}_${data}_ratio${sparsity_ratio}_type${sparsity_type}_nsamples${nsamples}_results.txt"

> $output_file

for seed in {0..4}; do
    seed_value=$((seed * 5))
    echo "Running with seed: $seed" >> $output_file
    echo "Running with calibration data: $data" >> $output_file
    CUDA_VISIBLE_DEVICES=1 python main_dclm_sample.py \
        --model $model \
        --prune_method sparsegpt \
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