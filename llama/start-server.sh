#!/usr/bin/env bash
# Qwen3.8 27B for FRIDAY: P100 (CUDA0) for the model, RX 580 (Vulkan0) for vision and MTP drafting.
# Two slots: 0 for phone chat, 1 for background work. Exact-prefix reuse plus saved slot
# snapshots are what make new chats start fast (see api/prompt_cache.py).
set -euo pipefail
: "${LLAMA_RUNTIME:?set in llama.env}" "${LLAMA_MODEL:?}" "${LLAMA_MMPROJ:?}" "${LLAMA_DRAFT:?}" "${LLAMA_SLOTS:?}"
unset QWENFAST_HALF2_EXACT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export QWENFAST_QK_SCALE_REUSE=3
export MTMD_BACKEND_DEVICE=Vulkan0
export LD_LIBRARY_PATH="$LLAMA_RUNTIME:/usr/local/cuda-12.9/lib64"
exec "$LLAMA_RUNTIME/llama-server" --model "$LLAMA_MODEL" --host 127.0.0.1 --port 8080 --alias qwen3.8-27b \
  --device CUDA0 --split-mode none --main-gpu 0 --gpu-layers all --ctx-size 65536 \
  --parallel 2 --kv-unified --cache-ram 8192 --no-cache-idle-slots --slot-save-path "$LLAMA_SLOTS" \
  --batch-size 2048 --ubatch-size 2048 --threads 8 --threads-batch 16 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 --fit off --metrics --no-webui --jinja \
  --mmproj "$LLAMA_MMPROJ" --image-max-tokens 1024 --reasoning-format deepseek --reasoning off --no-context-shift \
  --spec-type draft-mtp --spec-draft-device Vulkan0 --gpu-layers-draft all --spec-draft-n-max 2 --spec-draft-p-min 0.7 \
  --spec-draft-model "$LLAMA_DRAFT"
