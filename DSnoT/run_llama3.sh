model='/ossfs/workspace/common_base_model/Meta-Llama-3-8B-Instruct'
data='slimpajama'
output_file="llama3_8b_instruct_${data}_results.txt"


# 清空输出文件
> $output_file

for seed in {0..19}; do
    seed_value=$((seed * 5))
    echo "Running with seed: $seed" >> $output_file
    echo "Running with calibration data: $data" >> $output_file
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --model $model \
        --prune_method DSnoT \
        --seed $seed_value \
        --data $data \
        --initial_method wanda \
        --model_type llama \
        --nsamples 1024 \
        --sparsity_ratio 0.5 \
        --sparsity_type 4:8 \
        --max_cycle_time 50 \
        --update_threshold 0.1 \
        --pow_of_var_regrowing 1 \
        --eval_benchmark >> $output_file 2>&1
    echo "Finished running with seed: $seed" >> $output_file
    echo "------------------------" >> $output_file
done
