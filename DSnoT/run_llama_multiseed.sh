#!/bin/bash

model='path/to/Meta-Llama-3-8B'
#model='path/to/llama-v2-7b'
#model='path/to/llama-v2-13b'
data='slimpajama'
sparsity_ratio=0.6
sparsity_type='unstructured'
nsamples=128
output_file="llama2_7b_${data}_ratio${sparsity_ratio}_type${sparsity_type}_nsamples${nsamples}_results.txt"


> $output_file

for seed in {0..4}; do
    seed_value=$((seed * 5))
    echo "Running with seed: $seed" >> $output_file
    echo "Running with calibration data: $data" >> $output_file
    CUDA_VISIBLE_DEVICES=7 python main.py \
        --model $model \
        --prune_method DSnoT \
        --seed $seed_value \
        --data $data \
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