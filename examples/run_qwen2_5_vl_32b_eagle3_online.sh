#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Training helper for Qwen2.5-VL-32B EAGLE3 (defaults to TP=1 to fit on a single GPU)
NUM_GPUS=${1:-1}
TP_SIZE=${2:-$NUM_GPUS}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path Qwen/Qwen2.5-VL-32B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2_5_vl_32b_eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/allava4v_train.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen2.5-VL-32B-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --dist-timeout 360 \
    --chat-template qwen2-vl \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 802816
