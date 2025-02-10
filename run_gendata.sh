export CUDA_VISIBLE_DEVICES=0
#model='path/Meta-Llama-3-8B'
model="path/DCLM-7B"
python generate_data.py \
    --base_model $model \
    --data_path 'path/wikipedia_5k.json' \
