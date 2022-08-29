python main.py \
    --max_epochs=50 \
    --lr=0.001 \
    --data_root_dir /media/disk2/proj_embedding_aug/extracted_mag40x_patch256_fp/resnet50_trunc_pt_patch_features/ \
    --split_dir splits/sicapv2/ \
    --results_dir sicapv2/ \
    --csv_fpath datasets_csv/labels.csv \
    --reg_type=L1 \
    --batch_size=64