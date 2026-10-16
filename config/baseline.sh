#!/usr/bin/env bash
# NVIDIA's published DGX Spark recipe for Nemotron 3.5 Super VL.
# Source: the model card's "Quick Start" vLLM config (DGX Spark / GB10 only).
# This is the BASELINE measured against the tuned config.
set -eu

MODEL="${MODEL:-/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8}"
DRAFTER="${DRAFTER:-/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2}"
NAME="${NAME:-supervl-base}"
PORT="${PORT:-18300}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
sync; sleep 10

docker run -d --name "$NAME" --restart no --ipc host --gpus all \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -v "${HF_DIR:-$HOME/.cache/huggingface}:/hf" \
  -p "127.0.0.1:${PORT}:8000" \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" \
  --served-model-name nemotron-3.5-super-vl \
  --trust-remote-code \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.75 \
  --max-num-batched-tokens 8192 \
  --speculative_config.model "$DRAFTER" \
  --speculative_config.num_speculative_tokens 7 \
  --speculative_config.method dflash \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --enable-mamba-shared-prefix-checkpoint

echo "baseline launching (NVIDIA recipe). ~8-10 min to load."
echo "watch: docker logs -f $NAME"