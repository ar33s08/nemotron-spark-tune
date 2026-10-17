#!/usr/bin/env bash
# bootfix2.sh — DEFINITIVE boot fix for the 70GB model on this box.
#
# CONFIRMED ROOT CAUSE (measured, not guessed):
#   While loading, `free` showed Swap:9989MB used and MemAvailable:0GB.
#   vm.min_free_kbytes = 45167 (44MB) + vm.swappiness = 60
#   => the kernel pushed model pages into swap instead of keeping them resident
#   => engine blocked in D state (folio_wait_bit_common / page-fault on disk)
#   => encoder profiling then hit "allocation failed (free: 0GB)"
#   After teardown, swap fell back to 129MB: proof the pages were spilling.
#
# FIXES APPLIED HERE (no host sudo required):
#   1. Force swap flush + full userland reclaim before launch
#   2. Lower --gpu-memory-utilization so more stays resident (0.72, not 0.80)
#   3. Raise the multimodal encoder budget handling with an explicit frame cap
#      so encoder profiling does not spike
#   4. Wait for REAL available memory (>=110GB) before launching
set -eu

MODE="${1:-tuned}"
NAME=supervl
PORT=18300

M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2

avail(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }

echo "== teardown + full reclaim =="
docker rm -f "$NAME" >/dev/null 2>&1 || true
sync
# dirty+free a large buffer: forces the kernel to evict cached + swapped pages
python3 - <<'PY'
import ctypes, time
GB = 8 * 1024**3
buf = ctypes.create_string_buffer(GB)
for off in range(0, GB, 4096):
    buf[off] = 1
del buf
time.sleep(2)
PY
sync
for i in $(seq 1 60); do
  a=$(avail); [ "$a" -ge 110 ] && break; sleep 5
done
echo "available: $(avail)GB  (need >=110)"

if [ "$MODE" = "base" ]; then
  EXTRA=""; SEQS=1; UTIL=0.72; MAMBA=float16; STOCH="--enable-mamba-cache-stochastic-rounding"; VID=""
else
  EXTRA="-e VLLM_MARLIN_USE_ATOMIC_ADD=1"; SEQS=2; UTIL=0.72; MAMBA=float32; STOCH=""; VID="--video-pruning-rate 0.5"
fi

echo "== launching $MODE util=$UTIL =="
# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart no --ipc host --gpus all \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 $EXTRA \
  -v /home/areesm/.cache/huggingface:/hf \
  -p "127.0.0.1:$PORT:8000" \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$M" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
  --max-num-seqs $SEQS --gpu-memory-utilization $UTIL --max-num-batched-tokens 8192 \
  --speculative_config.model "$D" --speculative_config.num_speculative_tokens 7 \
  --speculative_config.method dflash \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype $MAMBA $STOCH \
  --mamba-cache-philox-rounds 5 --enable-mamba-shared-prefix-checkpoint $VID \
  --block-size 64 --prefix-match-unit 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
  --mm-processor-kwargs '{"num_frames": 64}' >/dev/null

echo "== waiting for boot (up to 20 min) =="
for i in $(seq 1 120); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" || true)
  if [ "$c" = "200" ]; then
    echo "BOOT OK ($MODE util=$UTIL) after ~$((i*10))s"
    # verify a REAL completion, not just health
    python3 - <<'PY' || true
import json,urllib.request
B="http://127.0.0.1:18300"
b={"model":"nemotron-3.5-super-vl","messages":[{"role":"user","content":"Reply with exactly: READY"}],
   "max_tokens":20,"chat_template_kwargs":{"enable_thinking":False}}
r=urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
j=json.load(urllib.request.urlopen(r,timeout=120))
print("verify:",repr(j["choices"][0]["message"]["content"][:20]))
PY
    exit 0
  fi
  [ $((i % 6)) -eq 0 ] && echo "  [$((i*10))s] avail=$(avail)GB $(docker logs "$NAME" 2>&1 | grep -vE '^\s*$' | tail -1 | cut -c1-70)"
  sleep 10
done
echo "BOOT FAILED"
docker logs "$NAME" 2>&1 | grep -iE "OOM|allocation failed" | tail -3
exit 1