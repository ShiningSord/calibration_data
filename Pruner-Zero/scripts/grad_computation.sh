# llama-2-7b
LLAMA2_PATH='/ossfs/workspace/common_base_model/Meta-Llama-3-8B-Instruct'
CUDA_VISIBLE_DEVICES=2 python lib/gradient_computation.py --nsamples 128 \
    --model $LLAMA2_PATH --llama_version 3 --task gradient
