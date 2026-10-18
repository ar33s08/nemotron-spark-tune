#!/usr/bin/env bash
# vidwall.sh — find the true video-on ceiling.
#
# Diagnosis from vidmid.sh: at ctx=393216 the failure was NOT the encoder OOM.
# It was KV starvation:
#   ValueError: max seq len (393216) needs 11.47 GiB KV, only 4.68 GiB available
# The encoder cache (== max_model_len whatever it is) eats the room, so KV
# starves. util 0.80 leaves too little.
#
# So: push util UP (more room -> bigger encoder + KV both fit), and walk ctx
# downward from 393216 to find the highest booting point. 262144 is known-OK at
# util 0.80, so also re-confirm it at the same util for a fair floor.
set -u
W=~/vidwall; mkdir -p "$W"; LOG="$W/w.log"
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

try(){ local maxlen="$1" util="$2"
  docker rm -f supervl >/dev/null 2>&1; sync
  for i in $(seq 1 60); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 113 ] && break; sleep 5; done
  log "### ctx=$maxlen util=$util VIDEO=1 (avail=${a}GB)"
  docker run -d --name supervl --restart no --ipc host --gpus all \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -v /home/areesm/.cache/huggingface:/hf -p 127.0.0.1:18300:8000 \
    vllm/vllm-openai:nightly-aarch64 \
    --model "$M" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
    --max-model-len "$maxlen" --max-num-seqs 2 --gpu-memory-utilization "$util" \
    --max-num-batched-tokens 8192 \
    --speculative_config.model "$D" --speculative_config.num_speculative_tokens 7 \
    --speculative_config.method dflash \
    --mamba-backend flashinfer --mamba-ssm-cache-dtype float32 \
    --mamba-cache-philox-rounds 5 --enable-mamba-shared-prefix-checkpoint \
    --video-pruning-rate 0.7 \
    --limit-mm-per-prompt '{"video":1,"image":1}' \
    --block-size 64 --prefix-match-unit 64 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 >/dev/null
  for i in $(seq 1 100); do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health)
    if [ "$c" = "200" ]; then
      kv=$(docker logs supervl 2>&1 | grep -o "GPU KV cache size: [0-9,]* tokens" | tail -1)
      log "ctx=$maxlen util=$util -> BOOT OK  ($kv)"
      echo "{\"ctx\":$maxlen,\"util\":$util,\"boot\":\"OK\",\"kv\":\"$kv\"}" >> "$W/res.json"; return 0
    fi
    docker ps --format '{{.Names}}' | grep -q supervl || break
    sleep 10
  done
  why=$(docker logs supervl 2>&1 | grep -oiE "needs [0-9.]+ GiB KV|allocation failed|only [0-9.]+ GiB available" | tail -1)
  log "ctx=$maxlen util=$util -> FAIL ($why)"
  echo "{\"ctx\":$maxlen,\"util\":$util,\"boot\":\"FAIL\",\"why\":\"$why\"}" >> "$W/res.json"; return 1
}

# 393216 failed at 0.80 (KV starved). Give it more room.
try 393216 0.88
try 393216 0.90
# if 393216 works, probe upward
try 524288 0.90
log "########## VIDWALL DONE ##########"