#!/bin/bash

export PJRT_DEVICE=TPU &&
export XLA_USE_BF16=0 &&
export PT_XLA_DEBUG=1 &&
export WANDB_RESUME="allow" &&
export WANDB_API_KEY="5b8322df11b04a8895325a5bf6ef8cbba0dd64a2" &&
export WANDB_ENTITY=conan1024hao &&
export WANDB_PROJECT=amasia &&

export MODEL_PATH="/mnt/disks/storage/llm_ckpts/Meta-Llama-3.1-8B-Instruct" &&
export DATA_PATH="/home/$USER/cambrian/Cambrian1k.jsonl" &&
export IMAGE_FOLDER="/mnt/disks/storage/data/finetune_data" &&
export CKPT_NAME="cambrian_test" &&
export CKPT_DIR="/home/$USER/$CKPT_NAME" &&

export TPU_PROCESS_BOUNDS=1,1,1
export TPU_VISIBLE_CHIPS=0

python cambrian/train/train_tpu.py \
    --model_name_or_path $MODEL_PATH \
    --version llama_v3 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower_aux_list '["siglip/CLIP-ViT-SO400M-14-384"]' \
    --vision_tower_aux_token_len_list '[576]' \
    --image_token_len 576 \
    --num_query_group 1 \
    --query_num_list '[576]' \
    --connector_depth 3 \
    --image_position 91 \
    --vision_hidden_size 1024 \
    --connector_only False \
    --num_of_vision_sampler_layers 10 \
    --start_of_vision_sampler_layers 0 \
    --stride_of_vision_sampler_layers 3 \
    --mm_projector_type sva \
    --mm_vision_sampler_lr 1e-4 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 False \
    --output_dir $CKPT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.06 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $CKPT_NAME \
    --fsdp "full_shard" \
    --fsdp_config fsdp_config.json
