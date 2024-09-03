CUDA_VISIBLE_DEVICES=1 python main.py \
    --model '/mntnlp/common_base_model/llama2-7b-chat' \
    --prune_method DSnoT \
    --initial_method wanda \
    --sparsity_ratio 0.3 \
    --sparsity_type unstructured \
    --max_cycle_time 50 \
    --update_threshold 0.1 \
    --pow_of_var_regrowing 1 \
    --save_model '/mntnlp/yixin.jyx/yixin.jyxx/yixin.jyx/output/llama2-7b-chat-DSnoT-wanda-0.3-unstructured'