# normal
bash pipeline.sh \
    --cuda_device 0 \
    --dagan_model mlp \
    --dagan_reg_type cosine \
    --dagan_epochs 200 \
    --dagan_batch_size 64 \
    --dagan_drop_out 0.2 \
    --dagan_learning_rate 0.001

# mlp
bash pipeline.sh \
    --cuda_device 0 \
    --dagan_model mlp \
    --dagan_reg_type None \

# transformer
bash pipeline.sh \
    --cuda_device 1 \
    --dagan_model transformer \
    --dagan_reg_type None \
    --dagan_n_heads 4 \
    --dagan_emb_dim 64 \

# independent
bash pipeline.sh \
    --cuda_device 2 \
    --dagan_model independent \
    --dagan_reg_type None \

# independent_fast
bash pipeline.sh \
    --cuda_device 2 \
    --dagan_model independent_fast \
    --dagan_reg_type None \