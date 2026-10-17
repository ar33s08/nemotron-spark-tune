#!/usr/bin/env bash
# ab-verify.sh — paired verification: NVIDIA recipe vs tuned config, back to back,
# same box, same harness. Runs baseline -> bench, tuned -> bench.
# Writes res-<label>.json for each. This is the honest apples-to-apples test.
set -u
W=~/abverify; mkdir -p "$W"; LOG="$W/ab.log"
MODEL=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
DRAFTER=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

launch(){ # $1 = mode: base|tuned
  docker rm -f supervl >/dev/null 2>&1; sync; sleep 10
  local extra="" mseq=1 util=0.75 mamba=float16 stoch="--enable-mamba-cache-stochastic-rounding" vid="" 
  if [ "$1" = "tuned" ]; then
    extra="-e VLLM_MARLIN_USE_ATOMIC_ADD=1"; mseq=2; util=0.80
    mamba=float32; stoch=""
    vid="--video-pruning-rate 0.5"
  fi
  # shellcheck disable=SC2086
  docker run -d --name supervl --restart no --ipc host --gpus all \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 $extra \
    -v /home/areesm/.cache/huggingface:/hf -p 127.0.0.1:18300:8000 \
    vllm/vllm-openai:nightly-aarch64 \
    --model "$MODEL" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
    --max-num-seqs $mseq --gpu-memory-utilization $util --max-num-batched-tokens 8192 \
    --speculative_config.model "$DRAFTER" --speculative_config.num_speculative_tokens 7 \
    --speculative_config.method dflash \
    --mamba-backend flashinfer --mamba-ssm-cache-dtype $mamba $stoch \
    --mamba-cache-philox-rounds 5 --enable-mamba-shared-prefix-checkpoint $vid \
    --block-size 64 --prefix-match-unit 64 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 >/dev/null
}
wait_up(){ for i in $(seq 1 100); do c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health); [ "$c" = "200" ] && return 0; sleep 10; done; return 1; }

bench(){ python3 - "$1" <<'PY'
import json,sys,urllib.request,time,threading,statistics as st
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
     "conc2":round(conc(2),1),"conc4":round(conc(4),1),
     "conc8":round(conc(8),1)}
out["mean"]=round((out["chat"]+out["code"])/2,1)
open(f"{__import__('os').path.expanduser('~')}/abverify/res-{L}.json","w").write(json.dumps(out))
print(json.dumps(out))
PY
}

for mode in base tuned; do
  log "### launching $mode"
  launch "$mode"
  if wait_up; then log "$mode UP"; bench "$mode" 2>&1 | tee -a "$LOG"; log "$mode DONE"; else log "$mode FAILED TO BOOT"; fi
done
log "################ AB VERIFY COMPLETE ################"