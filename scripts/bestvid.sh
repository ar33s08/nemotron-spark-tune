#!/usr/bin/env bash
# bestvid.sh — the definitive video-ON config: ctx 524288, util 0.90, benchmarked.
# This is the new "best model" config with video. Produce real numbers for it,
# then compare against the 1M video-off config.
set -u
W=~/bestvid; mkdir -p "$W"; LOG="$W/b.log"
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run(){ local tag="$1" maxlen="$2" util="$3" vid="$4"
  docker rm -f supervl >/dev/null 2>&1; sync
  for i in $(seq 1 60); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 113 ] && break; sleep 5; done
  log "### $tag ctx=$maxlen util=$util video=$vid (avail=${a}GB)"
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
    --limit-mm-per-prompt "{\"video\":$vid,\"image\":1}" \
    --block-size 64 --prefix-match-unit 64 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3)
  for i in $(seq 1 140); do
    [ "$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null)" != "true" ] && { log "$tag FAIL"; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health)" = "200" ] && break
    sleep 10
  done
  log "$tag BOOT OK  $(docker logs "$cid" 2>&1 | grep -o 'GPU KV cache size: [0-9,]* tokens' | tail -1)"
  python3 - "$tag" <<'PY' | tee -a "$LOG"
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
open(os.path.expanduser(f"~/bestvid/{L}.json"),"w").write(json.dumps(out)); print("RESULT",json.dumps(out))
PY
}

run "videoON-524288" 524288 0.90 1
run "videoOFF-1M"    1048576 0.85 0
log "########## BESTVID DONE ##########"