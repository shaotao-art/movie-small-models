#!/bin/bash
export MASTER_PORT=10001
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export NPROC_PER_NODE=8
if [ $NPROC_PER_NODE -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
else
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi


dataset_p="/apdcephfs_cq8/share_1534723/taoshao/wx-gui-agent/data/open-source-cleaned-data/grounding/grounding_pretrain_data_albu/sampled_amex_100k.json"
model_dir="/tmp/model_cache/Qwen/Qwen3.5-4B"
model=Qwen3.5-4B
run_name=test-sft-qwen3.5-4b
MAX_PIXELS=$((1024 * 32 * 32))
max_token_length=4096
global_batch_size=128
num_epoch=1
max_lr=5e-5
weight_decay=0.01
warmup_ratio=0.03
lr_scheduler_type="cosine"
lora_rank=8
lora_alpha=16
lora_dropout=0.05
per_device_train_batch_size=2

cur_date=$(date +%m-%d)
cur_time=$(date +%H-%M)
gradient_accumulation_steps=$((global_batch_size / (NPROC_PER_NODE * per_device_train_batch_size)))
base_dir="/apdcephfs_cq8/share_1534723/taoshao/rl-grounding/saves"
EXPERIMENT=${model}_${run_name}_${MAX_PIXELS}
LOG_ROOT=${base_dir}/output/${cur_date}/${model}/${EXPERIMENT}_${cur_time}
mkdir -p ${LOG_ROOT}
LOGGING_FILE=${LOG_ROOT}/run.log
swift sft \
    --model ${model_dir} \
    --dataset ${dataset_p} \
    --tuner_type lora \
    --output_dir ${LOG_ROOT} \
    --max_pixels ${MAX_PIXELS} \
    --max_length ${max_token_length} \
    --target_modules "all-linear" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --freeze_llm False \
    --freeze_vit False \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --dataloader_num_workers 4 \
    --lora_dtype bfloat16 \
    --torch_dtype bfloat16 \
    --num_train_epochs ${num_epoch} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --attn_impl flash_attn \
    --learning_rate ${max_lr} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed zero1 2>&1 | tee ${LOGGING_FILE}
