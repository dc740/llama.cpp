#!/usr/bin/env bash

export LLAMA_CACHE="./cache/unsloth/Qwen3.6-35B-A3B-GGUF"
./build/bin/llama-server \
    -m ~/Scaricati/models/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf \
    --temp 0.6 \
    --presence-penalty 0.3 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --cache-type-k bf16 --cache-type-v bf16 \
    --parallel 8 \
    --ctx-size 262144 --rope-scaling yarn --rope-scale 4 \
    --context-shift \
    --predict 65536 \
    --flash-attn on \
    --chat-template-kwargs '{"preserve_thinking":true}' \
    --kv-unified \
    --no-mmap

# For coding use temperature=0.6, presence-penalty=0.0
#     --presence_penalty 1.5 \
#     --temp 1.0 \
#    --ctx-size 262144 --rope-scaling yarn --rope-scale 4 \

