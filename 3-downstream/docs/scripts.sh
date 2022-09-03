# TODO: Fix batch_size issue when larger than 1
CUDA_DEVICES=0 python main.py \
    --data_root_dir /media/disk2/proj_embedding_aug/extracted_mag40x_patch256_fp/resnet50_trunc_pt_patch_features/ \
    --split_dir splits/sicapv2/ \
    --results_dir sicapv2/ \
    --csv_fpath datasets_csv/labels.csv \
    --num_runs 5 \
    --max_epochs 100 \
    --lr 0.001 \
    --batch_size=1 \
    --augmentation combined \
    --dagan