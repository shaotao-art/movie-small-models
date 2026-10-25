#!/bin/bash

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
model_ckp_path=/home/tao/movie-small-models/ckp-tag/get_sub_model-merged
vllm serve $model_ckp_path \
--served-model-name qwen35 \
--host 0.0.0.0 \
--port 8080 \
--data-parallel-size $NUM_GPUS \
--gpu-memory-utilization 0.6 \
--max-model-len 4096 \
--api-key dummy
