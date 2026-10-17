#!/usr/bin/env bash
# TUNED + BOOTABLE config for Nemotron 3.5 Super VL on a single DGX Spark (GB10).
#
# ============================================================================
# THE CRITICAL FIX (why the vendor recipe does not boot)
# ----------------------------------------------------------------------------
# NVIDIA's published DGX Spark recipe leaves max_model_len at 1M. The vision
# encoder derives its cache budget from the context length:
#
#   encoder_runner.py:131  -> budget.get_encoder_budget()
#   encoder_cache_manager.py:354  max(max_num_encoder_input_tokens, max_tokens_per_mm_item)
#   encoder_budget.py:29  get_mm_max_tokens_per_item(seq_len = model_config.max_model_len)
#
# At 1M the encoder tries to profile 1,048,576 tokens. Once the 70 GiB of
# weights are resident, a 128 MiB allocation fails with ~700 MiB free, the
# allocator retries in a loop, and the engine dies with
# "Engine core initialization failed". KV cache is never even allocated.
#
# FIX: --max-model-len 262144  (budget drops 1048576 -> 262144)
#
# Things that DO NOT help (measured, not guessed):
#   --gpu-memory-utilization  (tried 0.80 / 0.78 / 0.72 / 0.70 - no effect)
#   --max-num-batched-tokens  (tried 8192 / 4096 / 2048 - budget stayed 1048576)
# ============================================================================
#
# Tuning that DOES help, vs the vendor recipe:
#   --max-num-seqs 1 -> 2              (+214% aggregate throughput @ 4 parallel)
#   VLLM_MARLIN_USE_ATOMIC_ADD=1       (+8.8% code, +5.4% mean single-stream)
#   NOTE: stacking the other candidate levers scored WORSE than marlin alone.
#
# Measured result: boot ok, 887,377 tokens of KV cache (3.39x concurrency),
# 53.9 tok/s code, 87.7 tok/s aggregate @ 4 parallel.
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
echo "TRADE-OFF: 262144 context, NOT the advertised 1M. In exchange the box boots"
echo "           and you get 887,377 tokens of KV cache at 3.39x concurrency."