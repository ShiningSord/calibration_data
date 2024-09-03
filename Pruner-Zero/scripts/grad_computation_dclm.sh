MODEL_PATH='/ossfs/workspace/datacube-nas/yixin_llm/DCLM-7B'
CUDA_VISIBLE_DEVICES=2 python lib/gradient_computation_dclm.py --nsamples 128 \
    --model $MODEL_PATH --seed 0 --data wikitext2 --task gradient