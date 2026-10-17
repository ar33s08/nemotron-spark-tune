#!/usr/bin/env bash
# verify.sh — prove the two claims a stranger would check first:
#   1. config/baseline.sh  (vendor recipe)  -- DOES IT ACTUALLY FAIL?
#   2. config/tuned.sh     (our fix)        -- DOES IT ACTUALLY BOOT?
#
# Also runs a paired throughput comparison on the tuned config so the
# speed claim is measured, not asserted.
set -u
W=~/verify; mkdir -p "$W"; LOG="$W/v.log"
M=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/8a115b9d966adce8c059a7bcf47f49a1e57957d8
D=/hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

clean(){ docker rm -f supervl >/dev/null 2>&1; sync
  for i in $(seq 1 40); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 112 ] && { log "avail=${a}GB"; return; }; sleep 5; done; log "avail=${a}GB (low)"; }

wait_up(){ for i in $(seq 1 120); do c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18300/health); [ "$c" = "200" ] && return 0; sleep 10; done; return 1; }

launch(){ local mode="$1"
  if [ "$mode" = "baseline" ]; then
    # VENDOR recipe verbatim: seqs 1, util 0.75, full 1M ctx, video ENABLED
    EXTRA=""; SEQS=1; UTIL=0.75; MAMBA=float16; STOCH="--enable-mamba-cache-stochastic-rounding"; MM='{"video":1,"image":1}'
  else
    # OUR FIX: video disabled -> encoder budget collapses -> full 1M boots
    EXTRA="-e VLLM_MARLIN_USE_ATOMIC_ADD=1"; SEQS=2; UTIL=0.85; MAMBA=float32; STOCH=""; MM='{"video":0,"image":1}'
  fi
  # shellcheck disable=SC2086
  docker run -d --name supervl --restart no --ipc host --gpus all \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=20 $EXTRA \
    -v /home/areesm/.cache/huggingface:/hf -p 127.0.0.1:18300:8000 \
    vllm/vllm-openai:nightly-aarch64 \
    --model "$M" --served-model-name nemotron-3.5-super-vl --trust-remote-code \
    --max-model-len 1048576 \
    --max-num-seqs $SEQS --gpu-memory-utilization $UTIL --max-num-batched-tokens 8192 \
    --speculative_config.model "$D" --speculative_config.num_speculative_tokens 7 \
    --speculative_config.method dflash \
    --mamba-backend flashinfer --mamba-ssm-cache-dtype $MAMBA $STOCH \
    --mamba-cache-philox-rounds 5 --enable-mamba-shared-prefix-checkpoint \
    --limit-mm-per-prompt "$MM" \
    --block-size 64 --prefix-match-unit 64 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 >/dev/null
}

for mode in baseline tuned; do
  log "########## $mode ##########"
  clean
  launch "$mode"
  if wait_up; then
    log "$mode: BOOT OK"
    python3 - "$mode" <<'PY' 2>&1 | tee -a "$LOG"
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
     "conc2":round(conc(2),1),"conc4":round(conc(4),1)}
out["mean"]=round((out["chat"]+out["code"])/2,1)
open(os.path.expanduser(f"~/verify/r-{L}.json"),"w").write(json.dumps(out)); print(json.dumps(out))
PY
  else
    log "$mode: BOOT FAILED"
    docker logs supervl 2>&1 | grep -iE "budget of|allocation failed|ValueError|Engine core" | tail -3 | tee -a "$LOG"
  fi
done
log "########## VERIFY COMPLETE ##########"