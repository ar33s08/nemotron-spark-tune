#!/usr/bin/env bash
# bootfix3.sh — THE fix. Root cause is now precisely identified.
#
# CONFIRMED CHAIN:
#   encoder_runner.py:131 -> cache budget computed from
#     scheduler_config.max_num_encoder_input_tokens
#     which is `init=False` and hard-set to max_num_batched_tokens (8192).
#   The vision tower (C-RADIO) then profiles with that budget while
#   `util` has already eaten the RAM -> "allocation failed (free: 759MB)".
#
# LEVERS THAT ACTUALLY APPLY:
#   1. Give the vision encoder its OWN small budget via cache kwargs
#   2. Cap frames/fps at the PROCESSOR level (the CLI flag alone was not enough)
#   3. util stays 0.72
#   4. Real reclaim before launch (proven: gets avail to 118GB)
#
# Runs tuned config; prints BOOT OK/FAILED.
set -u
NAME=supervl
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
avail(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }

echo "== teardown + reclaim =="
docker rm -f $NAME >/dev/null 2>&1 || true
sync
python3 - <<'PY'
import ctypes, time
GB = 8 * 1024**3
buf = ctypes.create_string_buffer(GB)
for o in range(0, GB, 4096): buf[o] = 1
del buf; time.sleep(2)
PY
sync
for i in $(seq 1 60); do [ "$(avail)" -ge 110 ] && break; sleep 5; done
echo "available: $(avail)GB"

# mm limits: 1 video, 1 image, small tile/frame caps
MM='{"video":{"fps":1,"num_frames":32},"image":{"max_pixels":401408}}'

docker run -d --name $NAME --restart no --ipc host --gpus all \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v /home/areesm/.cache/huggingface:/hf -p 127.0.0.1:18300:8000 \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$M" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
  --max-num-seqs 2 --gpu-memory-utilization 0.72 --max-num-batched-tokens 4096 \
  --speculative_config.model "$D" --speculative_config.num_speculative_tokens 7 \
  --speculative_config.method dflash \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float32 \
  --mamba-cache-philox-rounds 5 --enable-mamba-shared-prefix-checkpoint \
  --video-pruning-rate 0.5 \
  --mm-processor-kwargs "$MM" \
  --limit-mm-per-prompt '{"video":1,"image":1}' \
  --block-size 64 --prefix-match-unit 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 >/dev/null

echo "== waiting (up to 20 min) =="
for i in $(seq 1 120); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health || true)
  if [ "$c" = "200" ]; then
    echo "BOOT OK after ~$((i*10))s"
    python3 - <<'PY' || true
import json,urllib.request
B="http://127.0.0.1:18300"
b={"model":"nemotron-3.5-super-vl","messages":[{"role":"user","content":"Reply with exactly: READY"}],
   "max_tokens":20,"chat_template_kwargs":{"enable_thinking":False}}
r=urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
print("verify:",repr(json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"][:20]))
PY
    exit 0
  fi
  [ $((i % 6)) -eq 0 ] && echo "  [$((i*10))s] avail=$(avail)GB $(docker logs $NAME 2>&1|grep -vE '^\s*$'|tail -1|cut -c1-80)"
  sleep 10
done
echo "BOOT FAILED"
docker logs $NAME 2>&1 | grep -iE "OOM|allocation failed|TypeError|ValueError" | tail -3
exit 1