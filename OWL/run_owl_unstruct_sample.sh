
model='path/to/Meta-Llama-3-8B'
#model='path/to/llama-v2-7b'
#model='path/to/llama-v2-13b'
data='sampled_wikipedia_selfgen_4_filter'
sampled_path='path/to/wikipedia_selfgen_4_filter.json'
sparsity_ratio=0.6
sparsity_type='unstructured'
sample_mode='selfgen'
output_file="llama3_8b_owl_${data}_ratio${sparsity_ratio}_type${sparsity_type}_nsamples${nsamples}_results.txt"

> $output_file

for seed in {0..19}; do
    seed_value=$((seed * 5))
    echo "Running with seed: $seed" >> $output_file
    echo "Running with calibration data: $data" >> $output_file
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --Lamda 0.08 \
        --Hyper_m 5 \
        --model $model \
        --data $data \
        --sampled_path $sampled_path \
        --sample_mode $sample_mode \
        --seed $seed \
        --prune_method wanda_owl \
        --sparsity_ratio ${sparsity_ratio} \
        --sparsity_type ${sparsity_type} \
        --eval_zero_shot >> $output_file 2>&1
    echo "Finished running with seed: $seed" >> $output_file
    echo "------------------------" >> $output_file
done