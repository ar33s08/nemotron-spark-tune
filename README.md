# Nemotron 3.5 Super VL on a single DGX Spark — it does not boot as shipped

**NVIDIA's published DGX Spark recipe for this model fails to start.** This repo
documents the root cause (traced to the source line), the one-flag fix, and the
tuning that follows once it boots.

Measured on **one** NVIDIA DGX Spark (GB10, 121 GB unified memory).

---

## TL;DR

```bash
# vendor recipe fails with: "Engine core initialization failed"
# fix = ONE flag:
--max-model-len 262144
```

Then it boots: 887,377 tokens of KV cache, 3.39x concurrency, 53.9 tok/s code,
87.7 tok/s aggregate at 4 parallel.

## The bug

NVIDIA's DGX Spark config leaves `max_model_len` at the model's advertised 1M.
The **vision encoder derives its cache budget from the context length**:

```
encoder_runner.py:131           -> budget.get_encoder_budget()
encoder_cache_manager.py:354    max(max_num_encoder_input_tokens, max_tokens_per_mm_item)
encoder_budget.py:29            get_mm_max_tokens_per_item(seq_len = model_config.max_model_len)
```

At 1M the encoder tries to profile **1,048,576 tokens**. Once the 70 GiB of
weights are resident, a 128 MiB allocation fails with ~700 MiB free, the
allocator spins in a retry loop, and the engine dies:

```
memory allocation failed with OOM on device 0 while trying to allocate 134217728 bytes
  (free: 740929536, total: 130663231488)
...
RuntimeError: Engine core initialization failed. See root cause above.
```

It never even reaches KV cache allocation.

### What does NOT fix it (all measured)

| attempt | result |
|---|---|
| `--gpu-memory-utilization` 0.80 → 0.78 → 0.72 → 0.70 | **no effect** |
| `--max-num-batched-tokens` 8192 → 4096 → 2048 | **no effect** (budget stayed 1048576) |
| `--mm-processor-kwargs '{"num_frames":64}'` | `TypeError` — not a real kwarg |
| `--mm-processor-kwargs '{"video":...}'` | `TypeError` — processor takes flat keys only |
| restarting / retrying | same failure every time |

The encoder budget is driven purely by `max_model_len`. Nothing else moves it.

### Two related landmines found on the way

1. **Swap spill.** Stock `min_free_kbytes` is ~44 MB and `swappiness` is 60, so the
   kernel pages the model out rather than keeping it resident. Proof: swap usage
   went **9989 MB → 129 MB** across a teardown. `config/tuned.sh` dirties and frees
   an 8 GB buffer before launching to force reclamation (no root needed).

2. **`--gpu-memory-utilization` is a SYSTEM setting on unified memory.** It is a
   slice of the whole 121 GB shared with the OS, not dedicated VRAM. At 0.85 it
   pins ~108 GB unswappable and the box wedges: `ping` works, TCP to :22 still
   *accepts*, but `sshd` never sends its banner because it cannot fork. Recovery
   needs a physical power button. GB10 unified pages are not reclaimed on their own.

## The trade-off, stated honestly

You get **262,144 context, not the advertised 1,000,000.** In exchange the box
boots, and you get 887,377 tokens of KV cache at **3.39x concurrency** — which is
what an agent workload actually uses.

## Tuning (once it boots)

Two changes vs the vendor recipe:

| flag | vendor | tuned | measured |
|---|---|---|---|
| `--max-num-seqs` | 1 | **2** | **+214% aggregate @ 4 parallel** |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | unset | **1** | **+8.8% code, +5.4% mean** |

> The `+214%` is a *concurrency* effect — `seqs=1` serialises parallel requests.
> Single-stream the honest number is +5.4% / +8.8%.

`VLLM_MARLIN_USE_ATOMIC_ADD` matters because GB10 has no native FP4, so every GEMM
routes through the Marlin kernel. Marlin/compile-layer knobs move throughput;
scheduling knobs barely move single-stream.

### Stacking the winners made it worse

| config | mean tok/s |
|---|---|
| `MARLIN_USE_ATOMIC_ADD` alone | **45.0** |
| both best levers | 43.1–43.8 |
| stock | 42.7 |

Individually-good flags collide. Ship one, not a pile.

## Full variant table (18 variants, 3 phases)

`base` = vendor recipe. Deltas vs that phase's baseline.
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

## Other failures, with verbatim errors

| attempt | error |
|---|---|
| drafter `candidate_topk` 8 → 16 | `AssertionError: Attempted to load weight (torch.Size([1,1,1,8,1024])) into parameter (torch.Size([1,1,1,16,1024]))` — vLLM source allows ≤16; the checkpoint's `rank_embedding` is baked at 8 |
| `--max-num-batched-tokens 16384` | won't boot: needs 12.82 GiB KV for 1M ctx, 6.66 available |
| `VLLM_BATCH_INVARIANT=1` | won't boot |
| `mamba-ssm-cache-dtype float32` + stochastic rounding | mutually exclusive: `Stochastic rounding for Mamba cache requires the SSM cache to be float16` |

## Other findings

- LiLiCorr spec depth is already maxed at 7 (its `block_size` is 8).
- FlashInfer's autotuner only completes a full pass on an idle box; under contention
  it silently fell back to `tactic=-1`. Those "no tuned config" warnings were a
  symptom of resource contention, not a broken kernel.
- `--kv-cache-dtype fp8` is a no-op: `fp8_e4m3` is already the default.

## Reproduce

```bash
bash config/baseline.sh   # vendor recipe (expected: fails to boot)
bash config/tuned.sh      # fixed + tuned (expected: boots in ~13 min)
python3 bench/bench.py --label tuned
```

Requires: docker, an NVIDIA DGX Spark, and the NVFP4 weights + LiLiCorr drafter
pre-downloaded under `HF_HOME`.

## Layout

```
config/baseline.sh   # NVIDIA's published DGX Spark recipe
config/tuned.sh      # the boot fix + the two tuning flags
bench/bench.py       # single-stream + concurrency benchmark (n=3 median)
```

## Honest limitations

- One machine, one model, one vendor's serving stack.
- The 1M context is *capped* to 262144. Whether 1M can be made to work by
  shrinking the encoder profile some other way is untested.
- ±1.5% run-to-run variance; effects below that are labelled flat.
- Rows from different phases ran on different box-idle conditions and are grouped
  accordingly rather than mixed into one headline.

## License

Apache-2.0

---

Model: [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) (OpenMDW-1.1)