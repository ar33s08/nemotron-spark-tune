#!/usr/bin/env bash
# edge.sh — settle the framing question with evidence, not opinion.
#
# Question: is the vendor recipe (a) BROKEN, or (b) just at the memory edge
# on a 121GB Spark, i.e. works if you give it room?
#
# Test: run the vendor recipe at 1M with video ON, sweeping util DOWNWARD.
# If SOME util boots it -> "works, but tight"  (NVIDIA not wrong)
# If NO util boots it   -> "cannot reach advertised 1M via config" (stronger)
#
# We already know 0.88/0.90/0.92 fail. Try LOW util, which frees memory.
set -u
W=~/edge; mkdir -p "$W"; LOG="$W/e.log"
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

try(){ local util="$1"
  docker rm -f supervl >/dev/null 2>&1; sync
  for i in $(seq 1 60); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 113 ] && break; sleep 5; done
  log "### VENDOR recipe 1M, video=1, util=$util  (avail=${a}GB)"
  docker run -d --name supervl --restart no --ipc host --gpus all \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 \
    -v /home/areesm/.cache/huggingface:/hf -p 127.0.0.1:18300:8000 \
    vllm/vllm-openai:nightly-aarch64 \
    --model "$M" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
    --max-model-len 1048576 --max-num-seqs 1 --gpu-memory-utilization "$util" \
    --max-num-batched-tokens 8192 \
    --speculative_config.model "$D" --speculative_config.num_speculative_tokens 7 \
    --speculative_config.method dflash \
    --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
    --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
    --enable-mamba-shared-prefix-checkpoint \
    --limit-mm-per-prompt '{"video":1,"image":1}' \
    --block-size 64 --prefix-match-unit 64 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 >/dev/null
  for i in $(seq 1 90); do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health)
    if [ "$c" = "200" ]; then log "util=$util -> BOOT OK"; docker logs supervl 2>&1|grep "GPU KV cache size"|tail -1|tee -a "$LOG"; return 0; fi
    docker ps --format '{{.Names}}' | grep -q supervl || break
    sleep 10
  done
  log "util=$util -> BOOT FAIL"; docker logs supervl 2>&1|grep -iE "allocation failed|Engine core|ValueError|budget of"|tail -2|tee -a "$LOG"
  return 1
}

# low util frees memory for weights; 0.60/0.70/0.80 all naive-fail per prior runs.
# The remaining untested lever: util low + video on is already 0.70-failed.
# So test the ACTUAL vendor default util too.
for u in 0.75 0.70 0.85; do try "$u"; done
log "########## EDGE DONE ##########"