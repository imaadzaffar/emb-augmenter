CUDA_DEVICES=2 python main.py \
    --results_dir sanity-check/ \
    --lr=0.001 \
    --dataset_size=100 \
    --batch_size=10 \
    --n_features=1024 \
    --drop_out=0.2 \
    --model_type=independent_fast