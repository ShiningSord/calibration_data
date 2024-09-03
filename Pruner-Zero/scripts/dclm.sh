#!/bin/bash

# Set common variables

sparsity_ratio=0.5


model="/ossfs/workspace/datacube-nas/yixin_llm/DCLM-7B"
gradient_path="/ossfs/workspace/datacube-nas/yixin_llm/gradients/dclm/gradients_aggregate_norm_l2_DCLM-7B_c4_10.path"


# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=2

# Define function to run python command
run_python_command() {
    python main_dclm.py \
        --model $model \
        --gradient_path $gradient_path \
        --prune_method $1 \
        --sparsity_ratio $sparsity_ratio \
        --sparsity_type $2 \
        --save_model $3 \
        --save $4 >./out/dclm_7b_instruct_${2}_${1}.txt
}

# # llama-7b with wanda pruning method
# echo "Running with wanda pruning method"
# CUDA_VISIBLE_DEVICES=1 run_python_command "wanda" "unstructured" "out/llama2_7b_chat/unstructured/wanda/"
# run_python_command "wanda" "2:4" "out/llama2_7b_chat/2-4/wanda/"
# run_python_command "wanda" "4:8" "out/llama2_7b_chat/4-8/wanda/"
# echo "Finished wanda pruning method"

# # llama-7b with sparsegpt pruning method
# echo "Running with sparsegpt pruning method"
# run_python_command "sparsegpt" "unstructured" "out/llama2_7b_chat/unstructured/sparsegpt/"
# run_python_command "sparsegpt" "2:4" "out/llama2_7b_chat/2-4/sparsegpt/"
# run_python_command "sparsegpt" "4:8" "out/llama2_7b_chat/4-8/sparsegpt/"
# echo "Finished sparsegpt pruning method"

# # llama-7b with magnitude pruning method
# echo "Running with magnitude pruning method"
# run_python_command "magnitude" "unstructured" "out/llama2_7b_chat/unstructured/magnitude/"
# run_python_command "magnitude" "2:4" "out/llama2_7b_chat/2-4/magnitude/"
# run_python_command "magnitude" "4:8" "out/llama2_7b_chat/4-8/magnitude/"
# echo "Finished magnitude pruning method"

# llama-7b with pruner-zero pruning method
echo "Running with pruner-zero pruning method"
run_python_command "pruner-zero" "unstructured" "/ossfs/workspace/yixin.jyx/output/dclm-7b-prunezero-0.5-c4-seed10" "out/dclm_7b/unstructured/pruner-zero/"
#run_python_command "pruner-zero" "2:4" "out/llama2_7b_chat/2-4/pruner-zero/"
#run_python_command "pruner-zero" "2:4" "/ossfs/workspace/yixin.jyx/output/dclm-7b-prunezero-2:4-wikipedia-seed10" "out/dclm_7b/2-4/pruner-zero/"
echo "Running with pruner-zero pruning method"
