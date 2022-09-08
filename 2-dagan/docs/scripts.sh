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
    --model_type=mlp

CUDA_DEVICES=1 python main.py \
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
    --model_type=transformer \
    --n_heads=2 \
    --emb_dim=32

CUDA_DEVICES=2 python main.py \
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
    --model_type=transformer \
    --n_heads=4 \
    --emb_dim=64


CUDA_DEVICES=1 python main.py \
    --data_root_dir /media/disk2/proj_embedding_aug/extracted_mag40x_patch256_fp/resnet50_trunc_pt_patch_features/ \
    --csv_fpath datasets_csv/labels.csv \
    --split_dir splits/sicapv2/ \
    --results_dir sicapv2/ \
    --max_epochs=200 \
    --lr=0.001 \
    --batch_size=64 \
    --drop_out=0.2 \
    --early_stopping \
    --reg_type=cosine \
    --model_type=transformer \
    --n_heads=2 \
    --emb_dim=2


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
    --model_type=transformer \
    --n_heads=4 \
    --emb_dim=64