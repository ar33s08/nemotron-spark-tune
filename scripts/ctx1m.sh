#!/usr/bin/env bash
# ctx1m.sh — can we get the ADVERTISED 1M context to boot?
#
# The encoder budget is derived from max_model_len. At 1M the profile is huge and
# OOMs. But the profile input is driven by mm_max_toks_per_item, which depends on
# the processor's dummy-input sizing. So: keep max_model_len=1M (full context) and
# shrink what the processor generates for the PROFILE instead.
#
# Levers tried in order:
#   A. 1M ctx + --max-num-batched-tokens high + limit-mm 1 + small video profile
#   B. 1M ctx + max_num_tiles=1 (fewer vision tiles -> smaller profile)
#
# Prints BOOT OK / BOOT FAILED with the encoder budget line, so we learn either way.
set -u
NAME=supervl
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
avail(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }

try(){
  local label="$1" maxlen="$2" util="$3"; shift 3
  echo "=== ATTEMPT: $label (maxlen=$maxlen util=$util) ==="
  docker rm -f $NAME >/dev/null 2>&1 || true
  sync
  python3 - <<'PY'
import ctypes,time
gb=8*1024**3
b=ctypes.create_string_buffer(gb)
for o in range(0,gb,4096): b[o]=1
del b; time.sleep(2)
PY
  sync
  for _ in $(seq 1 50); do [ "$(avail)" -ge 112 ] && break; sleep 5; done
  echo "avail=$(avail)GB"

  # shellcheck disable=SC2086
  docker run -d --name $NAME --restart no --ipc host --gpus all \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -v /home/areesm/.cache/huggingface:/hf -p 127.0.0.1:18300:8000 \
    vllm/vllm-openai:nightly-aarch64 \
    --model "$M" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
    --max-model-len "$maxlen" \
    --max-num-seqs 2 --gpu-memory-utilization "$util" --max-num-batched-tokens 8192 \
    --speculative_config.model "$D" --speculative_config.num_speculative_tokens 7 \
    --speculative_config.method dflash \
    --mamba-backend flashinfer --mamba-ssm-cache-dtype float32 \
    --mamba-cache-philox-rounds 5 --enable-mamba-shared-prefix-checkpoint \
    --video-pruning-rate 0.7 \
    --limit-mm-per-prompt '{"video":1,"image":1}' \
    $* \
    --block-size 64 --prefix-match-unit 64 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 >/dev/null

  for i in $(seq 1 120); do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health || true)
    if [ "$c" = "200" ]; then
      echo "RESULT: BOOT OK  [$label]  after ~$((i*10))s"
      docker logs $NAME 2>&1 | grep -iE "budget of|GPU KV cache size" | tail -2
      python3 - <<'PY' || true
import json,urllib.request
B="http://127.0.0.1:18300"
b={"model":"nemotron-3.5-super-vl","messages":[{"role":"user","content":"Reply: OK"}],
   "max_tokens":10,"chat_template_kwargs":{"enable_thinking":False}}
r=urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
print("verify:",repr(json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"][:20]))
PY
      return 0
    fi
    [ $((i % 10)) -eq 0 ] && echo "  [$((i*10))s] avail=$(avail)GB"
    sleep 10
  done
  echo "RESULT: BOOT FAILED  [$label]"
  docker logs $NAME 2>&1 | grep -iE "budget of|ValueError|OOM|allocation failed" | tail -3
  return 1
}

echo "########## TEST 1: full 1M context ##########"
try "1M-ctx" 1048576 0.88 --mm-processor-kwargs '{"max_num_tiles":1}'
echo
echo "########## TEST 2: 1M ctx, util 0.92 ##########"
try "1M-ctx-util92" 1048576 0.92 --mm-processor-kwargs '{"max_num_tiles":1}'
echo
echo "=== ending state ==="
curl -s -o /dev/null -w "health=%{http_code}\n" --max-time 5 http://127.0.0.1:18300/health || true