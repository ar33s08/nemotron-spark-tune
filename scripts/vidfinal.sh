#!/usr/bin/env bash
# vidfinal.sh — correct, race-free probe of the highest video-on context.
#
# Bugs fixed from vidtop.sh:
#  1. readback raced: an old server still answering /health made the new boot
#     look "OK" instantly, and a slow load looked "FAIL". Now we wait for the
#     container to EXIT (fail) or a NEW server to answer (pass), by pid/uptime.
#  2. util 0.92 was untested while a load was in flight; retry cleanly.
#
# Established facts:
#   video ON, ctx 524288, util 0.90 -> BOOT OK, 601,993 tok KV   (verified)
#   video ON, ctx 1048576 -> fails at every util (0.60..0.92)
#   video OFF, ctx 1048576 -> works, 2,012,851 tok KV            (shipped config)
set -u
W=~/vidfinal; mkdir -p "$W"; LOG="$W/f.log"
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

bench(){ python3 - "$1" <<'PY'
import json,sys,urllib.request,time,threading,statistics as st,os
L=sys.argv[1]; B="http://127.0.0.1:18300"
def run(p,mt,n=3):
    o=[]
    for _ in range(n):
        b={"model":"nemotron-3.5-super-vl","messages":[{"role":"user","content":p}],"max_tokens":mt,"chat_template_kwargs":{"enable_thinking":False}}
        t0=time.time();r=urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
        j=json.load(urllib.request.urlopen(r,timeout=300));o.append(j.get("usage",{}).get("completion_tokens",0)/max(time.time()-t0,1e-9))
    return st.median(o)
def conc(n):
    res={}
    def one(i):
        b={"model":"nemotron-3.5-super-vl","messages":[{"role":"user","content":"Count slowly from 1 to 40, one per line."}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":False}}
        t0=time.time();r=urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
        j=json.load(urllib.request.urlopen(r,timeout=300));res[i]=(j.get("usage",{}).get("completion_tokens",0),time.time()-t0)
    th=[threading.Thread(target=one,args=(i,)) for i in range(n)];t0=time.time()
    [t.start() for t in th];[t.join() for t in th]
    return sum(v[0] for v in res.values())/max(time.time()-t0,1e-9)
out={"label":L,"chat":round(run("Explain what a hash table is.",150),1),
     "code":round(run("Write a python function that merges two sorted lists.",200),1),
     "conc4":round(conc(4),1)}
out["mean"]=round((out["chat"]+out["code"])/2,1)
open(os.path.expanduser(f"~/vidfinal/{L}.json"),"w").write(json.dumps(out)); print("BENCH",json.dumps(out))
PY
}

try(){ local maxlen="$1" util="$2"
  docker rm -f supervl >/dev/null 2>&1; sync
  for i in $(seq 1 60); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 113 ] && break; sleep 5; done
  log "### ctx=$maxlen util=$util VIDEO=1 (avail=${a}GB)"
  local cid
  cid=$(docker run -d --name supervl --restart no --ipc host --gpus all \
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
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3) || { log "docker run failed"; return 1; }
  # race-free: poll container state + health; only "up" counts if it is THIS cid
  for i in $(seq 1 140); do
    running=$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null || echo false)
    if [ "$running" != "true" ]; then
      why=$(docker logs "$cid" 2>&1 | grep -oiE "needs [0-9.]+ GiB KV|allocation failed to allocate|Free memory on device[^.]*|estimated maximum model length is [0-9]+" | tail -1)
      log "ctx=$maxlen util=$util -> FAIL: ${why:-exited}"
      echo "{\"ctx\":$maxlen,\"util\":$util,\"boot\":\"FAIL\",\"why\":\"${why:-exited}\"}" >> "$W/res.json"; return 1
    fi
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health)
    if [ "$c" = "200" ]; then
      log "ctx=$maxlen util=$util -> BOOT OK, benchmarking..."
      kv=$(docker logs "$cid" 2>&1 | grep -o "GPU KV cache size: [0-9,]* tokens" | tail -1)
      log "  $kv"
      echo "{\"ctx\":$maxlen,\"util\":$util,\"boot\":\"OK\",\"kv\":\"$kv\"}" >> "$W/res.json"
      bench "$maxlen-$util" | tee -a "$LOG"
      return 0
    fi
    sleep 10
  done
  log "ctx=$maxlen util=$util -> TIMEOUT"; return 1
}

# util 0.90 is proven at 524288. Probe upward with a clean, race-free harness.
try 786432 0.90
try 1048576 0.90
log "########## VIDFINAL DONE ##########"