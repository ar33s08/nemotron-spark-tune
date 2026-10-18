#!/usr/bin/env bash
# vidmid.sh — find the HIGHEST context that still boots with VIDEO ON.
#
# Why: the encoder cache budget == max_model_len when video is on
# (nano_nemotron_vl.py:361-363, via encoder_budget.py:29 seq_len=max_model_len).
# 1M fails. 262144 works. The whole 262K..1M range is UNTESTED.
#
# Goal: keep video AND as much context as possible. If e.g. 524288 works with
# video, that is strictly better than the current "262K or no video" choice.
#
# util is pinned at 0.80 (1M-failure was not util-sensitive, so hold it).
set -u
W=~/vidmid; mkdir -p "$W"; LOG="$W/v.log"
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

try(){ local maxlen="$1"
  docker rm -f supervl >/dev/null 2>&1; sync
  for i in $(seq 1 60); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 113 ] && break; sleep 5; done
  log "### VIDEO ON, ctx=$maxlen  (avail=${a}GB)"
  docker run -d --name supervl --restart no --ipc host --gpus all \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -v /home/areesm/.cache/huggingface:/hf -p 127.0.0.1:18300:8000 \
    vllm/vllm-openai:nightly-aarch64 \
    --model "$M" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
    --max-model-len "$maxlen" --max-num-seqs 2 --gpu-memory-utilization 0.80 \
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
      log "ctx=$maxlen -> BOOT OK  ($kv)"; echo "{\"ctx\":$maxlen,\"video\":1,\"boot\":\"OK\",\"kv\":\"$kv\"}" >> "$W/res.json"; return 0
    fi
    docker ps --format '{{.Names}}' | grep -q supervl || break
    sleep 10
  done
  log "ctx=$maxlen -> FAIL"
  echo "{\"ctx\":$maxlen,\"video\":1,\"boot\":\"FAIL\"}" >> "$W/res.json"; return 1
}

# bisect the untested middle. 262144 known-OK, 1048576 known-FAIL.
for ml in 524288 786432 393216; do try "$ml"; done
log "########## VIDMID DONE ##########"