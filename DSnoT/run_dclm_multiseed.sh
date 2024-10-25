#!/bin/bash


model='/path/to/DCLM-7B'
data='slimpajama'
sparsity_ratio=0.65
sparsity_type='unstructured'
nsamples=128
output_file="dclm_7b_${data}_ratio${sparsity_ratio}_type${sparsity_type}_nsamples${nsamples}_results.txt"


> $output_file

for seed in {0..19}; do
    seed_value=$((seed * 5))
    echo "Running with seed: $seed" >> $output_file
    echo "Running with calibration data: $data" >> $output_file
    CUDA_VISIBLE_DEVICES=0 python main_dclm.py \
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