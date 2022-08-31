CUDA_DEVICES=0 python main.py \
    --data_root_dir /media/disk2/proj_embedding_aug/extracted_mag40x_patch256_fp/resnet50_trunc_pt_patch_features/ \
    --csv_fpath datasets_csv/labels.csv \
    --split_dir splits/sicapv2/ \
    --results_dir sicapv2/ \
    --max_epochs=50 \
    --lr=0.001 \
    --reg_type=L1 \
    --batch_size=64
    --model_type=Transformer

CUDA_DEVICES=0 python main.py \
    --data_root_dir /media/disk2/proj_embedding_aug/extracted_mag40x_patch256_fp/resnet50_trunc_pt_patch_features/ \
    --csv_fpath datasets_csv/labels.csv \
    --split_dir splits/sicapv2/ \
    --results_dir sicapv2/ \
    --max_epochs=200 \
    --lr=0.001 \
    --batch_size=64 \
    --drop_out=0.2 \
    --early_stopping \
    --reg_type=L1 \
    --model_type=Transformer