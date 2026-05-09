worker_count=$1
OUTPUT_DIR=$2
DATA_INFO=$3
BASE_DIR=$4

gradacc_steps=4
lr=5e-6
max_train_steps=10000
checkpointing_steps=20
lr_schedule="constant"
optimizer="adamw"
lr_warmup_steps=10
beta_dpo=5000
sft_lambda=0.1
dpo_lambda=1.0
mask_dpo_lambda=1.0
sft_lambda_mask=0.1

model_args="\
    --wanx_pretrained_model_name_or_path ${BASE_DIR} \
    --wanx_ref_model_name_or_path ${BASE_DIR} \
    --lora_rank 64 --lora_alpha 128 \
"

data_args="\
    --data_infos ${DATA_INFO} \
    --dataloader_num_workers 1 \
    --shuffle \
    --pin_memory \
    --buckets_config configs.buckets.480px_mf_mask \
    --frame_sampling_mode interval \
    --min_step 1 \
    --max_step 2 \
    --caption_keys gen_caption \
    --caption_weights 1.0 \
    --drop_last \
    --disable_fps_prefix \
"

log_args="\
    --report_to tensorboard \
    --nccl_timeout 7200 \
    --log_base ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
"

train_args="\
    --seed 42 \
    --mixed_precision bf16 \
    --train_batch_size 1 \
    --max_train_steps ${max_train_steps} \
    --checkpointing_steps ${checkpointing_steps} \
    --gradient_accumulation_steps ${gradacc_steps} \
    --gradient_checkpointing \
    --learning_rate ${lr} \
    --lr_scheduler ${lr_schedule} \
    --lr_warmup_steps ${lr_warmup_steps} \
    --lr_num_cycles 1 \
    --enable_slicing \
    --enable_tiling \
    --optimizer ${optimizer} \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.001 \
    --max_grad_norm 1.0 \
    --allow_tf32 \
    --num_train_epochs 10 \
    --beta_dpo ${beta_dpo} \
    --sft_lambda ${sft_lambda} \
    --dpo_lambda ${dpo_lambda} \
    --mask_dpo_lambda ${mask_dpo_lambda} \
    --sft_lambda_mask ${sft_lambda_mask} \
"

args="--num_processes ${worker_count} \
    innerT2V/train_wanx21.py ${model_args} ${data_args} ${log_args} ${train_args}"

echo "Args is: ${args}"
