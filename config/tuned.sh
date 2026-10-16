#!/usr/bin/env bash
# TUNED config for Nemotron 3.5 Super VL on a single DGX Spark (GB10).
#
# Two-flag diff vs NVIDIA's published recipe:
#   1. --max-num-seqs 1 -> 2                  (parallel requests scale instead of queue)
#   2. VLLM_MARLIN_USE_ATOMIC_ADD=1           (GB10 has no native FP4 -> Marlin path win)
#
# Measured: mean +5.4%, code +8.8%, aggregate@4parallel +214% vs the stock recipe.
# NOTE: stacked combos (4 or 2 levers) both scored WORSE than marlin alone. Ship one.
set -eu

MODEL="${MODEL:-/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8}"
DRAFTER="${DRAFTER:-/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2}"
NAME="${NAME:-supervl}"
PORT="${PORT:-18300}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
sync; sleep 10

docker run -d --name "$NAME" --restart no --ipc host --gpus all \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v "${HF_DIR:-$HOME/.cache/huggingface}:/hf" \
  -p "127.0.0.1:${PORT}:8000" \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" \
  --served-model-name nemotron-3.5-super-vl \
  --trust-remote-code \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 8192 \
  --speculative_config.model "$DRAFTER" \
  --speculative_config.num_speculative_tokens 7 \
  --speculative_config.method dflash \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float32 \
  --mamba-cache-philox-rounds 5 \
  --enable-mamba-shared-prefix-checkpoint \
  --video-pruning-rate 0.5 \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --block-size 64 \
  --prefix-match-unit 64 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3

echo "tuned config launching. ~8-10 min to load."
echo "watch: docker logs -f $NAME"
echo
echo "SAFETY: --gpu-memory-utilization is a slice of UNIFIED memory on GB10."
echo "        0.85 wedges the box (unswappable, sshd cannot fork). Keep <=0.80."