#!/usr/bin/env bash
# Start Nemotron 3.5 Super VL on one DGX Spark (GB10) at full 1M context.
#
# ============================================================================
# THE FIX
# ----------------------------------------------------------------------------
# The shipped config leaves max_model_len at the model's advertised 1M. When
# video input is enabled the vision encoder sets its cache budget to the ENTIRE
# context, so it profiles 1,048,576 tokens:
#
#   nano_nemotron_vl.py:361-363
#       if mm_counts.get("video", 0) > 0:
#           mm_max_tokens["video"] = seq_len      # seq_len == max_model_len
#
# That budget does not fit next to the ~70 GiB of resident weights on a 121 GB
# box. A 128 MiB allocation fails with ~700 MiB free, the allocator retries, and
# the engine dies with "Engine core initialization failed". KV cache is never
# even reached.
#
# FIX: --limit-mm-per-prompt '{"video":0,"image":1}'   (keeps all 1M context)
#      video off routes the encoder through the image-sizing path instead.
#
# Need video frames? MAXLEN=262144 VIDEO=1 also starts.
#
# Things that DO NOT help (measured, not guessed):
#   --gpu-memory-utilization  (0.60/0.70/0.75/0.85/0.90/0.92 - all fail)
#   --max-num-batched-tokens  (8192/4096/2048 - budget stayed 1048576)
#
# Tuning that DOES help:
#   --max-num-seqs 1 -> 2              (+56% aggregate throughput @ 4 parallel)
#   VLLM_MARLIN_USE_ATOMIC_ADD=1       (+8.8% code, +5.4% mean single-stream)
#   NOTE: stacking other candidate levers scored WORSE than marlin alone.
#
# Measured result: starts, 1,048,576 context, 1,958,130 tokens of KV cache.
# ============================================================================
set -eu

MODEL="${MODEL:-/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8}"
DRAFTER="${DRAFTER:-/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2}"
NAME="${NAME:-supervl}"
PORT="${PORT:-18300}"
MAXLEN="${MAXLEN:-1048576}"
UTIL="${UTIL:-0.85}"
# Video OFF = full 1M context. Set VIDEO=1 (and MAXLEN=262144) if you need frames.
VIDEO="${VIDEO:-0}"

# --- step 1: reclaim memory BEFORE launching ---------------------------------
# min_free_kbytes is ~44MB on a stock Spark and swappiness is 60, so the kernel
# spills model pages to swap instead of keeping them resident. Drop caches
# needs root; dirtying+freeing a big buffer is the userland equivalent.
# Proof this matters: swap went 9989MB -> 129MB after teardown.
docker rm -f "$NAME" >/dev/null 2>&1 || true
sync
python3 - <<'PY'
import ctypes, time
gb = 8 * 1024**3
buf = ctypes.create_string_buffer(gb)
for off in range(0, gb, 4096):
    buf[off] = 1
del buf
time.sleep(2)
PY
sync
for _ in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 110 ] && break
  sleep 5
done
echo "memory available before launch: ${avail}GB (want >=110)"

# --- step 2: launch ----------------------------------------------------------
docker run -d --name "$NAME" --restart no --ipc host --gpus all \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v "${HF_DIR:-$HOME/.cache/huggingface}:/hf" \
  -p "127.0.0.1:${PORT}:8000" \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" \
  --served-model-name nemotron-3.5-super-vl \
  --trust-remote-code \
  --max-model-len "$MAXLEN" \
  --max-num-seqs 2 \
  --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens 8192 \
  --speculative_config.model "$DRAFTER" \
  --speculative_config.num_speculative_tokens 7 \
  --speculative_config.method dflash \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float32 \
  --mamba-cache-philox-rounds 5 \
  --enable-mamba-shared-prefix-checkpoint \
  --video-pruning-rate 0.7 \
  --limit-mm-per-prompt "{\"video\":$VIDEO,\"image\":1}" \
  --block-size 64 \
  --prefix-match-unit 64 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3

echo "launching (max-model-len=$MAXLEN util=$UTIL). ~8-10 min to load."
echo "watch: docker logs -f $NAME"
echo
echo "context=$MAXLEN  video=$VIDEO  util=$UTIL"
echo "1M context needs VIDEO=0 (video on caps the encoder budget at the full context)."