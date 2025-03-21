#!/bin/bash

export WANDB_PROJECT=VoCo_Llama
export WANDB_NAME=vicuna-7b-v1.5-instruct-tuning

deepspeed --include localhost:0,1,2,3 --master_port=25600 llava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /volsparse3/wxd/models/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /volsparse3/wxd/models/llava_v1_5_mix665k.json \
    --image_folder /data/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /volsparse3/wxd/models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/voco_llama_ckpt \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
