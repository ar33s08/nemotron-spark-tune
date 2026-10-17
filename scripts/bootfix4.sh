#!/usr/bin/env bash
# bootfix4.sh — DEFINITIVE. Root cause fully traced to source.
#
# COMPLETE CHAIN (all verified in the container's source):
#   encoder_runner.py:131  prints "budget of 1048576 tokens"
#     -> budget.get_encoder_budget()
#     -> encoder_cache_manager.py:354  max(max_num_encoder_input_tokens, max_tokens_per_mm_item)
#     -> scheduler.py:303  max_num_encoder_input_tokens = max_num_batched_tokens   [init=False]
#     -> encoder_budget.py:29  max_tokens_per_item = processor.info.get_mm_max_tokens_per_item(
#                                 seq_len = model_config.max_model_len)      <-- 1M !!
#   vision_config: max_num_patches=13312, cls_token_per_teacher=true, 3 teachers, image_size 224
#   => the profiled video item is huge AND the budget is derived from 1M ctx
#   => encoder profile asks for ~128MB with only ~700MB free -> OOM loop, no KV cache ever allocated
#
# THE FIX: cap max_model_len so the encoder budget is bounded. 1M ctx is not needed
# for the encoder's *profile*; we keep 1M available via the model but bound the
# scheduler's encoder inputs by lowering max-num-batched-tokens AND giving a small
# max-model-len for profiling headroom.
set -u
NAME=supervl
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
avail(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }

UTIL="${1:-0.70}"
MNL="${2:-}"          # optional max-model-len cap
EXTRA_MNL=""
[ -n "$MNL" ] && EXTRA_MNL="--max-model-len $MNL"

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
echo "available: $(avail)GB  util=$UTIL  ${EXTRA_MNL:-(full 1M ctx)}"

# shellcheck disable=SC2086
docker run -d --name $NAME --restart no --ipc host --gpus all \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_MEDIA_LOADING_THREAD_COUNT=4 \
  -v /home/areesm/.cache/huggingface:/hf -p 127.0.0.1:18300:8000 \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$M" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
  $EXTRA_MNL \
  --max-num-seqs 2 --gpu-memory-utilization $UTIL --max-num-batched-tokens 2048 \
  --speculative_config.model "$D" --speculative_config.num_speculative_tokens 7 \
  --speculative_config.method dflash \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float32 \
  --mamba-cache-philox-rounds 5 --enable-mamba-shared-prefix-checkpoint \
  --video-pruning-rate 0.7 \
  --limit-mm-per-prompt '{"video":1,"image":1}' \
  --block-size 64 --prefix-match-unit 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 >/dev/null

echo "== waiting (up to 20 min) =="
for i in $(seq 1 120); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health || true)
  if [ "$c" = "200" ]; then
    echo "BOOT OK after ~$((i*10))s  util=$UTIL ${EXTRA_MNL}"
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
  [ $((i % 6)) -eq 0 ] && echo "  [$((i*10))s] avail=$(avail)GB $(docker logs $NAME 2>&1|grep -vE '^\s*$'|tail -1|cut -c1-78)"
  sleep 10
done
echo "BOOT FAILED (util=$UTIL ${EXTRA_MNL})"
docker logs $NAME 2>&1 | grep -iE "budget of|OOM|allocation failed|KV cache size" | tail -4
exit 1