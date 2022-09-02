
#--------------- SCRIPT ---------------

sicapv2_dagan_pipeline \
    --cuda_device 0 \
    --data_root_dir /media/disk2/proj_embedding_aug/extracted_mag40x_patch256_fp/resnet50_trunc_pt_patch_features/ \
    --batch_size 64 \
    --dagan_epochs 200 \
    --dagan_batch_size 64 \
    --dagan_lr 0.001 \
    --dagan_drop_out 0.2 \
    --dagan_model_type transformer \
    --dagan_reg_type cosine
    --dagan_n_heads 4 \
    --dagan_emb_dim 64 \
    --dagan_early_stopping \
    --classifier_epochs 100 \
    --classifier_batch_size 64 \
    --classifier_lr 0.001 \


#--------------- STEPS ---------------

1. start training GAN with hyperparameters from args
CUDA_DEVICES=[cuda_device] python 2-dagan/main.py \
    --data_root_dir [data_root_dir] \
    --csv_fpath datasets_csv/labels.csv \
    --split_dir splits/sicapv2/ \
    --results_dir sicapv2/ \
    --max_epochs [dagan_epochs] \
    --lr=0.001 \
    --batch_size [dagan_batch_size] \
    --drop_out [dagan_drop_out] \
    --early_stopping \
    --model_type [dagan_model_type] \
    --reg_type [dagan_reg_type] \
    --n_heads [dagan_n_heads] \
    --emb_dim [dagan_emb_dim] \

2. once finished -> somehow return param_code / results_dir after finishing

3. start classification training, load generator from saved model
CUDA_DEVICES=[cuda_device] python 3-downstream/main.py \
    --data_root_dir [data_root_dir] \
    --split_dir splits/sicapv2/ \
    --results_dir sicapv2/ \
    --csv_fpath datasets_csv/labels.csv \
    --num_runs 5 \
    --max_epochs [classifier_epochs] \
    --lr [classifier_lr] \
    --batch_size [classifier_batch_size] \
    --augmentation combined \
    --dagan \
    --dagan_src results/sicapv2/[dagan_param_code]/model_G_[dagan_param_code].txt \
