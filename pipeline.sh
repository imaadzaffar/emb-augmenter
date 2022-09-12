#!/usr/bin/env python

# -------------------------------- ARGS ------------------------------------

DAGAN_N_HEADS=4
DAGAN_EMB_DIM=64

while [ $# -gt 0 ] ; do
  case $1 in
    --cuda_device) CUDA_DEVICE="$2" ;;
    --dagan_model) DAGAN_MODEL="$2" ;;
    --dagan_n_heads) DAGAN_N_HEADS="$2" ;;
    --dagan_emb_dim) DAGAN_EMB_DIM="$2" ;;
    --dagan_reg_type) DAGAN_REG_TYPE="$2" ;;
    --dagan_batch_size) DAGAN_BATCH_SIZE="$2" ;;
    --dagan_epochs) DAGAN_EPOCHS="$2" ;;
    --dagan_learning_rate) DAGAN_LEARNING_RATE="$2" ;;
    --dagan_drop_out) DAGAN_DROP_OUT="$2" ;;

  esac
  shift
done

echo "dagan settings"
echo "--model: $DAGAN_MODEL"
echo "--n_heads: $DAGAN_N_HEADS"
echo "--emb_dim: $DAGAN_EMB_DIM"
echo "--reg_type: $DAGAN_REG_TYPE"

# -------------------------------- VARIABLES ------------------------------------

# files/directories
DATA_ROOT_DIR="/media/disk2/proj_embedding_aug/extracted_mag40x_patch256_fp/resnet50_trunc_pt_patch_features/"
CSV_FPATH="datasets_csv/labels.csv"
SPLIT_DIR="splits/sicapv2/"
RESULTS_DIR="sicapv2/"

# dagan
DAGAN_EPOCHS=200
DAGAN_BATCH_SIZE=64
DAGAN_LEARNING_RATE=0.001
DAGAN_DROP_OUT=0.2

# classifier
CLASSIFIER_AUGMENTATION=combined
CLASSIFIER_EPOCHS=100
CLASSIFIER_BATCH_SIZE=64
CLASSIFIER_LEARNING_RATE=0.001
CLASSIFIER_RUNS=5
CLASSIFIER_DROP_OUT=0.2

# -------------------------------- COMMANDS ------------------------------------

# dagan
cd 2-dagan

echo
echo "Started training DA-GAN..."

DAGAN_RUN_CODE=$(
  CUDA_DEVICES=$CUDA_DEVICE python main.py \
    --data_root_dir $DATA_ROOT_DIR \
    --csv_fpath $CSV_FPATH \
    --split_dir $SPLIT_DIR \
    --results_dir $RESULTS_DIR \
    --max_epochs $DAGAN_EPOCHS \
    --batch_size $DAGAN_BATCH_SIZE \
    --drop_out $DAGAN_DROP_OUT \
    --lr $DAGAN_LEARNING_RATE \
    --model_type $DAGAN_MODEL \
    --n_heads $DAGAN_N_HEADS \
    --emb_dim $DAGAN_EMB_DIM \
    --reg_type $DAGAN_REG_TYPE \
    --early_stopping
)

echo "Finished training DA-GAN: $DAGAN_RUN_CODE"


# classifier
cd ../3-downstream

echo 
echo "Started training classifier..."

RES_2=$(
  CUDA_DEVICES=$CUDA_DEVICE python main.py \
    --data_root_dir $DATA_ROOT_DIR \
    --split_dir $SPLIT_DIR \
    --results_dir $RESULTS_DIR \
    --csv_fpath $CSV_FPATH \
    --num_runs $CLASSIFIER_RUNS \
    --max_epochs $CLASSIFIER_EPOCHS \
    --batch_size $CLASSIFIER_BATCH_SIZE \
    --lr $CLASSIFIER_LEARNING_RATE \
    --augmentation $CLASSIFIER_AUGMENTATION \
    --dagan_run_code $DAGAN_RUN_CODE \
    --dagan_model $DAGAN_MODEL \
    --dagan_n_heads $DAGAN_N_HEADS \
    --dagan_emb_dim $DAGAN_EMB_DIM \
)

echo "Finished training classifier"