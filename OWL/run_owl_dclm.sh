
model='path/to/DCLM-7B'
data='wikipedia'
sparsity_ratio=0.6
sparsity_type='unstructured'
nsamples=128
output_file="dclm_7b_owl_${data}_ratio${sparsity_ratio}_type${sparsity_type}_nsamples${nsamples}_results.txt"

> $output_file

for seed in {0..19}; do
    seed_value=$((seed * 5))
    echo "Running with seed: $seed" >> $output_file
    echo "Running with calibration data: $data" >> $output_file
    CUDA_VISIBLE_DEVICES=3 python main_dclm.py \
        --Lamda 2 \
        --Hyper_m 6 \
        --model $model \
        --data $data \
        --seed $seed \
        --prune_method wanda_owl \
        --sparsity_ratio ${sparsity_ratio} \
        --sparsity_type ${sparsity_type} \
        --eval_zero_shot >> $output_file 2>&1
    echo "Finished running with seed: $seed" >> $output_file
    echo "------------------------" >> $output_file
done