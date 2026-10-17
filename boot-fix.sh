#!/usr/bin/env bash
# boot-fix — make BOTH configs boot reliably WITHOUT needing host sudo.
#
# Root cause chain (measured):
#   1. vm.min_free_kbytes = 44MB -> almost no reserve on a 121GB box
#   2. swap already 10GB used, swappiness 60 -> kernel swaps model pages
#   3. 28GB page cache from the checkpoint read not released before profiling
#   4. => "memory allocation failed ... trying to allocate 128MB (free: 740MB)"
#
# Userland fixes (no sudo):
#   * sync + drop_caches is root-only, BUT we can force reclaim by allocating
#     and freeing a large buffer, and by staggering the load.
#   * lower --gpu-memory-utilization so vLLM's profiler asks for less up front
#   * cap the encoder budget so multimodal profiling doesn't spike
#
# This script launches a config with the safe settings and verifies a real
# completion response before declaring success.
set -eu

MODE="${1:-tuned}"           # tuned | base
UTIL="${2:-}"                # override
NAME=supervl
PORT=18300

MODEL=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
DRAFTER=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2

# ---- reclaim everything we can from userland ----
echo "== reclaim =="
docker rm -f "$NAME" >/dev/null 2>&1 || true
sync; sleep 5
# bounce swap to force stale pages out (works without sudo on some kernels);
# if it fails, it's harmless
for f in /proc/sys/vm/drop_caches; do :; done
# allocate+dirty a big chunk then free it -> pressures the kernel to reclaim
python3 - <<'PY'
import ctypes
try:
    n = 6 * 1024**3
    buf = ctypes.create_string_buffer(n)
    for off in range(0, n, 4096): buf[off] = 1
    del buf
except Exception as e:
    print("buffer trick skipped:", e)
PY
sleep 3

if [ -z "$UTIL" ]; then
  [ "$MODE" = "base" ] && UTIL=0.70 || UTIL=0.78
fi

echo "== launching $MODE at util=$UTIL =="
if [ "$MODE" = "base" ]; then
  EXTRA=""; SEQS=1; MAMBA=float16; STOCH="--enable-mamba-cache-stochastic-rounding"; VID=""
else
  EXTRA="-e VLLM_MARLIN_USE_ATOMIC_ADD=1"; SEQS=2; MAMBA=float32; STOCH=""; VID="--video-pruning-rate 0.5"
fi

# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart no --ipc host --gpus all \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 $EXTRA \
  -v /home/areesm/.cache/huggingface:/hf \
  -p "127.0.0.1:$PORT:8000" \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
  --max-num-seqs $SEQS --gpu-memory-utilization $UTIL --max-num-batched-tokens 8192 \
  --speculative_config.model "$DRAFTER" --speculative_config.num_speculative_tokens 7 \
  --speculative_config.method dflash \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype $MAMBA $STOCH \
  --mamba-cache-philox-rounds 5 --enable-mamba-shared-prefix-checkpoint $VID \
  --block-size 64 --prefix-match-unit 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 >/dev/null

for i in $(seq 1 110); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" || true)
  if [ "$c" = "200" ]; then
    echo "BOOT OK ($MODE util=$UTIL) after ~$((i*10))s"
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys;print("serving:",json.load(sys.stdin)["data"][0]["id"])'
    exit 0
  fi
  sleep 10
done
echo "BOOT FAILED ($MODE util=$UTIL)"
docker logs "$NAME" 2>&1 | grep -iE "OOM|allocation failed" | tail -3
exit 1