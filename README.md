# Nemotron 3.5 Super VL on a single DGX Spark (GB10, 121 GB)

**NVIDIA's published DGX Spark recipe did not boot on my Spark at its advertised
1M context.** This repo shows what happened, traces it to a specific line in the
model source, gives the one-flag fix, and reports what it measures like after.

Measured on **one** DGX Spark. If yours behaves differently, that is useful — the
reproduction script is one file.

![benchmark](assets/results.png)

---

## TL;DR

```bash
# On a 121 GB Spark, the published recipe at 1M exits with:
#   RuntimeError: Engine core initialization failed
#
# FIX = disable video (keeps your FULL 1M context):
--limit-mm-per-prompt '{"video":0,"image":1}'
```

**Why:** when video is enabled, the vision encoder sets its cache budget to the
**entire context length** (`nano_nemotron_vl.py:361-363`). At 1M it profiles
1,048,576 tokens, and that budget does not fit alongside the ~70 GiB of resident
weights on a 121 GB box. Disabling video routes the encoder through the
image-sizing path instead.

**Result:** boots at **full 1,048,576 context**, 1,958,130 tokens of KV cache.

> Need video? Use `--max-model-len 262144` instead. Also boots, 887,377 tokens of
> KV cache, but you give up 75% of context.

## Root cause

The published config leaves `max_model_len` at the model's advertised 1M.

```
nano_nemotron_vl.py:361-363
    if mm_counts.get("video", 0) > 0:
        mm_max_tokens["video"] = seq_len        # seq_len == max_model_len
encoder_budget.py:29   -> get_mm_max_tokens_per_item(seq_len = max_model_len)
encoder_runner.py:131  -> "Encoder cache will be initialized with a budget of 1048576 tokens"
```

The engine confirms the budget itself, on the way down:

```
Encoder cache will be initialized with a budget of 1048576 tokens,
and profiled with 1 video items of the maximum feature size.
```

1,048,576 == `max_model_len`. Once the weights are resident, a 128 MiB allocation
fails with ~700 MiB free, the allocator retries, the engine dies. It never reaches
KV cache allocation:

```
memory allocation failed with OOM on device 0 while trying to allocate 134217728 bytes
  (free: 740929536, total: 130663231488)
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

### Not a tuning problem — every `util` fails

Six values of `--gpu-memory-utilization`, video on, 1M, clean box each run:

| util | result |
|---|---|
| 0.60 | fail |
| 0.70 | fail |
| 0.75 | fail |
| 0.85 | fail |
| 0.90 | fail |
| 0.92 | fail |

`0.92` fails *earlier* and differently — before loading:
`Free memory on device (109.5 GiB) < desired (111.95 GiB)`.

The encoder budget is driven purely by `seq_len` while video is on. Only two
things move it: **lower the context**, or **remove video from the mix**.

### What does NOT fix it (all measured)

| attempt | result |
|---|---|
| `--gpu-memory-utilization` sweep (6 values) | **no effect** |
| `--max-num-batched-tokens` 8192 → 4096 → 2048 | **no effect** (budget stayed 1048576) |
| `--max-num-batched-tokens 16384` | won't boot: needs 12.82 GiB KV, 6.66 available |
| `--mm-processor-kwargs '{"num_frames":64}'` | `TypeError` — not a real kwarg |
| `--mm-processor-kwargs '{"video":...}'` | `TypeError` — flat keys only |
| `--mm-processor-kwargs '{"max_num_tiles":1,"video_pruning_rate":0.9}'` | still OOMs |
| restarting / retrying | same failure every time |

### The two working fixes

| goal | flag | result |
|---|---|---|
| **full 1M context** | `--limit-mm-per-prompt '{"video":0,"image":1}'` | boots, 1,958,130 tok KV |
| keep video | `--max-model-len 262144` | boots, 887,377 tok KV |

## Results: vendor recipe vs fixed config

One script run (`scripts/basenum.sh`), n=3 median, clean box each leg.

| metric | vendor recipe @262K\* | ours @262K | **ours @1M** |
|---|---|---|---|
| chat tok/s | **37.7** | 31.6 | 32.0 |
| code tok/s | 54.3 | **54.9** | 52.8 |
| single-stream mean | **46.0** | 43.2 | 42.4 |
| 2 parallel (agg) | 60.1 | **81.3** | **83.7** |
| **4 parallel (agg)** | 60.2 | **85.4** | **94.1** |

\* The vendor recipe cannot boot at 1M, so 262K is the nearest point where it runs.

### Read this honestly

- **The vendor recipe is ~8% FASTER single-stream** (46.0 vs 42.4). `seqs=1` does
  less scheduler work per request. A real trade, not a win.
- **Ours is +56% under parallel load** (94.1 vs 60.2) — that is the `seqs 2`
  change: parallel requests scale instead of serialising.
- **Ours serves 1M context. Theirs serves nothing at 1M.**

One request at a time? Their config is genuinely fine, once you cap the context so
it boots. Fan out, or need 1M? Ours is the only one that runs.

### Stacking the winners made it worse

| config | mean tok/s |
|---|---|
| `VLLM_MARLIN_USE_ATOMIC_ADD` alone | **45.0** |
| both best levers | 43.1–43.8 |
| stock | 42.7 |

Individually-good flags collide. Ship one, not a pile. `MARLIN_USE_ATOMIC_ADD`
matters because GB10 has no native FP4, so every GEMM routes through Marlin.

## Two related landmines

1. **Swap spill.** Stock `min_free_kbytes` is ~44 MB and `swappiness` is 60, so the
   kernel pages the model out instead of keeping it resident. Proof: swap went
   **9989 MB → 129 MB** across a teardown. `config/tuned.sh` dirties and frees an
   8 GB buffer before launching to force reclamation (no root needed).

2. **`--gpu-memory-utilization` is a SYSTEM setting on unified memory.** It is a
   slice of the whole 121 GB shared with the OS, not dedicated VRAM. At 0.85 it
   pinned ~108 GB unswappable and wedged the box: `ping` works, TCP to :22 still
   *accepts*, but `sshd` never sends its banner because it cannot fork. Recovery
   needs a physical power button.

## Full variant table (18 variants, 3 phases)

`base` = vendor recipe at that phase's settings. Run-to-run variance on this box is
**±1.5%** — anything inside that is called flat.

### Phase 1 — scheduling (baseline: mean 38.6 / conc4 27.3)

| variant | mean | code | conc4 | verdict |
|---|---|---|---|---|
| **seqs2 (`--max-num-seqs 2`)** | 39.0 | 57.1 | **37.9** | **WIN** |
| baseline | 38.6 | 57.3 | 27.3 | — |
| async8 (`--async-scheduling`) | 39.5 | 57.0 | 29.4 | flat |
| kvfp8 (`--kv-cache-dtype fp8`) | 38.3 | 56.6 | 28.1 | flat (fp8 already default) |
| async + batch16k | 28.2 | 36.9 | 26.7 | **−27%** |

### Phase 2 — kernel / compile layer (baseline: mean 42.7 / conc4 88.5)

| variant | mean | code | conc4 | verdict |
|---|---|---|---|---|
| **marlin_atomic_add** | **45.0** | **55.7** | 85.2 | **WIN** |
| cudagraph_trim | 44.3 | 54.1 | 87.9 | +3.7% |
| jitwarmup_threads | 44.1 | 53.0 | 85.9 | +3.3% |
| autotune_cache | 43.6 | 53.4 | **94.2** | +6.4% conc4 |
| moe_skip_padding | 42.8 | 53.9 | 87.1 | flat |
| baseline | 42.7 | 51.2 | 88.5 | — |
| max_tokens_expert | 42.0 | 52.4 | 84.4 | −1.6% |
| cudagraph_gc | 40.9 | 48.9 | 89.3 | −4.2% |

### Phase 3 — stacking (baseline: mean 42.7)

| config | mean | verdict |
|---|---|---|
| marlin alone | **45.0** | best |
| combo_all (4 levers) | 43.8 | worse |
| combo_safe (2 levers) | 43.1 | worse |

## Other failures, with verbatim errors

| attempt | error |
|---|---|
| drafter `candidate_topk` 8 → 16 | `AssertionError: Attempted to load weight (torch.Size([1,1,1,8,1024])) into parameter (torch.Size([1,1,1,16,1024]))` — vLLM source allows ≤16; the checkpoint's `rank_embedding` is baked at 8 |
| `VLLM_BATCH_INVARIANT=1` | won't boot |
| `mamba-ssm-cache-dtype float32` + stochastic rounding | mutually exclusive: `Stochastic rounding for Mamba cache requires the SSM cache to be float16` |

## Other findings

- LiLiCorr spec depth is already maxed at 7 (its `block_size` is 8).
- FlashInfer's autotuner only completes a full pass on an idle box; under contention
  it silently fell back to `tactic=-1`. The "no tuned config" warnings were a
  symptom of contention, not a broken kernel.
- `--kv-cache-dtype fp8` is a no-op: `fp8_e4m3` is already the default.

## Reproduce

```bash
bash verify.sh                    # both legs, paired A/B, records the failure
bash scripts/basenum.sh           # the three measured legs behind the table
bash scripts/edge.sh              # the 6-value util sweep
bash config/tuned.sh              # just the working config
python3 bench/bench.py --label tuned
```

Requires: docker, a DGX Spark, and the NVFP4 weights + LiLiCorr drafter
pre-downloaded under `HF_HOME`.

## Layout

```
verify.sh            # paired A/B: vendor vs fixed, back to back
config/baseline.sh   # NVIDIA's published DGX Spark recipe
config/tuned.sh      # the fix (video=0) + tuning flags
bench/bench.py       # single-stream + concurrency benchmark (n=3 median)
scripts/             # the individual experiments, kept for the record
```

## Honest limitations

- One machine, one model, one serving stack. A different Spark may differ.
- Video-off is a **behaviour change**, not a free win: no video understanding at 1M.
  The 262K+video config exists for that.
- ±1.5% run-to-run variance; effects below that are labelled flat.
- Phase tables ran under different box-idle conditions, so they are grouped rather
  than mixed into one headline number.

## License

Apache-2.0

---

Model: [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) (OpenMDW-1.1)