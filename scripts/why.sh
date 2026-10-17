#!/usr/bin/env bash
# why.sh — answer "was there no other solution to the context cut?"
#
# ROOT CAUSE (now fully proven from source):
#   nano_nemotron_vl.py:361-363
#       if mm_counts.get("video", 0) > 0:
#           mm_max_tokens["video"] = seq_len     # <-- the ENTIRE context length
#   So with video enabled, the encoder cache budget == max_model_len. At 1M that is
#   a 1M-token profile and the box has no room.
#
# THE UNTESTED ALTERNATIVE: disable video entirely (--limit-mm-per-prompt video=0).
#   Then mm_counts["video"]==0, the encoder sizes from images instead, and the
#   budget should collapse -> full 1M context may fit.
#
# Trade: no video understanding, but full 1M text/image context.
set -u
NAME=supervl
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
avail(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }

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
echo "avail=$(avail)GB  --> launching 1M ctx with video DISABLED"

docker run -d --name $NAME --restart no --ipc host --gpus all \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v /home/areesm/.cache/huggingface:/hf -p 127.0.0.1:18300:8000 \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$M" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
  --max-model-len 1048576 \
  --max-num-seqs 2 --gpu-memory-utilization 0.85 --max-num-batched-tokens 8192 \
  --speculative_config.model "$D" --speculative_config.num_speculative_tokens 7 \
  --speculative_config.method dflash \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float32 \
  --mamba-cache-philox-rounds 5 --enable-mamba-shared-prefix-checkpoint \
  --limit-mm-per-prompt '{"video":0,"image":1}' \
  --block-size 64 --prefix-match-unit 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 >/dev/null

for i in $(seq 1 120); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health || true)
  if [ "$c" = "200" ]; then
    echo "RESULT: BOOT OK with FULL 1M CONTEXT (video disabled)"
    docker logs $NAME 2>&1 | grep -iE "budget of|GPU KV cache size" | tail -2
    python3 - <<'PY' || true
import json,urllib.request
B="http://127.0.0.1:18300"
print("ctx:",json.load(urllib.request.urlopen(B+"/v1/models"))["data"][0].get("max_model_len"))
b={"model":"nemotron-3.5-super-vl","messages":[{"role":"user","content":"Reply: OK"}],
   "max_tokens":10,"chat_template_kwargs":{"enable_thinking":False}}
r=urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
print("verify:",repr(json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"][:20]))
PY
    exit 0
  fi
  [ $((i % 10)) -eq 0 ] && echo "  [$((i*10))s] avail=$(avail)GB"
  sleep 10
done
echo "RESULT: BOOT FAILED (video disabled, 1M ctx)"
docker logs $NAME 2>&1 | grep -iE "budget of|ValueError|allocation failed" | tail -3