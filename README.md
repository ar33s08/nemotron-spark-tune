# Nemotron 3.5 Super VL on a single DGX Spark — tuning findings

Measured on **one** NVIDIA DGX Spark (GB10, 121 GB unified memory), serving
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` with vLLM.

NVIDIA publishes a DGX Spark config for this model. This repo is what happened when
I benchmarked that recipe and then tried to beat it — including everything that
**didn't** work.

**18 variants tested. 3 phases. Two changes shipped.**

---

## Results: NVIDIA's recipe vs tuned

Same box, same prompts, same session, n=3 median, `enable_thinking=false`.

| metric | NVIDIA recipe | tuned | delta |
|---|---|---|---|
| code generation | 51.2 tok/s | **55.7 tok/s** | **+8.8%** |
| single-stream mean | 42.7 tok/s | **45.0 tok/s** | **+5.4%** |
| aggregate @ 4 parallel | 27.3 tok/s | **85.7 tok/s** | **+214%** |

> **Read the caveat.** The +214% is a *concurrency* effect: NVIDIA's recipe pins
> `--max-num-seqs 1`, so parallel requests serialise. Single-stream the honest gain
> is +5.4% / +8.8%. Both numbers are real — quote the one matching your workload.

## What shipped (the whole diff)

| flag | NVIDIA recipe | tuned |
|---|---|---|
| `--max-num-seqs` | `1` | **`2`** |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | *(unset)* | **`1`** |

Everything else in `config/tuned.sh` is either NVIDIA's own value or a safety fix.

**Why these two:** on GB10 there is no native FP4, so every GEMM routes through the
Marlin kernel — Marlin/compile-layer knobs move throughput, scheduling knobs barely
move single-stream. And `seqs=1` was leaving parallel throughput on the floor.

## The counterintuitive finding

Stacking individually-winning levers made things **worse** — two independent combos
both lost to the single best lever:

| config | mean tok/s |
|---|---|
| `MARLIN_USE_ATOMIC_ADD` alone | **45.0** |
| 4 levers stacked | 43.8 |
| 2 best levers stacked | 43.1 |
| stock baseline | 42.7 |

Individually-good flags collide. **Ship one, not a pile.**

## Full variant table (all 18, measured)

`base` = NVIDIA recipe. Bold = shipped. Deltas vs that phase's baseline.
Run-to-run variance on this box is **±1.5%** — anything inside that is called flat.

### Phase 1 — scheduling (baseline: mean 38.6 / conc4 27.3)

| variant | mean | code | conc4 | verdict |
|---|---|---|---|---|
| **seqs2 (`--max-num-seqs 2`)** | 39.0 | 57.1 | **37.9** | **WIN** |
| baseline | 38.6 | 57.3 | 27.3 | — |
| async8 (`--async-scheduling`) | 39.5 | 57.0 | 29.4 | flat |
| kvfp8 (`--kv-cache-dtype fp8`) | 38.3 | 56.6 | 28.1 | flat (fp8 is already default) |
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

## What failed, with verbatim errors

Every one of these was tested, not assumed.

| attempt | result |
|---|---|
| drafter `candidate_topk` 8 → 16 | `AssertionError: Attempted to load weight (torch.Size([1,1,1,8,1024])) into parameter (torch.Size([1,1,1,16,1024]))` — vLLM source says ≤16 is legal; the checkpoint's `rank_embedding` is baked at 8 |
| `--max-num-batched-tokens 16384` | won't boot: needs 12.82 GiB KV for 1M ctx, 6.66 available |
| `--async-scheduling` + batch 16384 | −27% throughput |
| `VLLM_BATCH_INVARIANT=1` | won't boot (incompatible with spec-decode path) |
| `--mamba-ssm-cache-dtype float32` + stochastic rounding | mutually exclusive: `Stochastic rounding for Mamba cache requires the SSM cache to be float16` |
| `--gpu-memory-utilization 0.85` | wedges the box — see below |

## The lesson worth the day: `gpu_memory_utilization` is a SYSEM setting on unified memory

On a DGX Spark, `gpu_memory_utilization` is a slice of the whole 128 GB **unified**
memory — shared with the OS — not dedicated VRAM. At `0.85` it pinned ~108 GB
unswappable and the box wedged:

- `ping` still worked
- TCP to port 22 still **accepted** (kernel-side, needs no allocation)
- `sshd` **never sent its banner** — it couldn't fork

Remote recovery was impossible. Physical power-button press required. GB10 unified
pages are not reclaimed without a reboot.

**Rule: keep it ≤0.80 and leave the OS room to breathe.**

## Other findings

- The video-encoder OOM is real; needs `--video-pruning-rate` + frame caps via
  `--media-io-kwargs`, which NVIDIA's pared-down recipe omits.
- LiLiCorr spec depth is already maxed at 7 (its `block_size` is 8).
- FlashInfer's autotuner only completes a full pass on an idle box; under contention
  it silently fell back to `tactic=-1`. The "no tuned config" warnings were a symptom
  of resource contention, not a broken kernel.
- `--kv-cache-dtype fp8` is a no-op: `fp8_e4m3` is already the default.

## Reproduce

```bash
# launch tuned config
bash config/tuned.sh

# launch NVIDIA's published recipe (baseline)
bash config/baseline.sh

# benchmark whatever is on :18300
python3 bench/bench.py --label tuned
```

Requires: docker, an NVIDIA DGX Spark, and the NVFP4 weights + LiLiCorr drafter
pre-downloaded under `HF_HOME`.

## Layout

```
config/baseline.sh   # NVIDIA's published DGX Spark recipe
config/tuned.sh      # the two-flag diff
bench/bench.py       # single-stream + concurrency benchmark (n=3 median)
docs/FINDINGS.md     # long-form write-up
```

## Honest limitations

- Single machine, single model, one vendor's serving stack.
- ±1.5% run-to-run variance; effects below that are labelled flat.
- Rows from different phases ran on different box-idle conditions and are grouped
  accordingly rather than mixed into one headline.

## License

Apache-2.0

---

Model: [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) (OpenMDW-1.1)