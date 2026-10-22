#!/usr/bin/env bash

export LLAMA_CACHE="./cache/unsloth/Qwen3.6-35B-A3B-GGUF"
./build/bin/llama-cli \
    -m ~/Scaricati/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --presence_penalty 1.5 \
    --min-p 0.00

# For coding use temperature=0.6, presence-penalty=0.0
