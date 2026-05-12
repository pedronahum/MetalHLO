# nanoGPT on tinyshakespeare — WIP, blocked on a backend bug

A 6-layer / 384-dim / 6-head decoder-only transformer
(~10.8 M parameters, ~150 GFLOPs/step) trained char-level on
tinyshakespeare with `batch_size=16`, `block_size=256`. Intended as the
compute-bound counterpart to
[`../karpathy_gpt/`](../karpathy_gpt) — same model family,
1000× more work per step. Should put MetalHLO's transformer-side
fusion infrastructure (LayerNorm fusion, attention fusion) through a
real workload.

## Status

**Currently broken on MetalHLO** with a backend bug that survived a
long bisect chain. The benchmark works on JAX CPU. Two compiler bugs
were found and fixed while debugging this — see the commit
`5ad3b92 Fix two compiler bugs surfaced by transformer workloads` —
and there's a third one remaining.

The minimal remaining repro (`debug_ln_attn.py` case C vs case A):

```python
# Case A: works
ln_out = jax.jit(manual_ln)(x, gamma, beta)   # materialised on device
attn_out = jax.jit(attn)(ln_out, w_qkv)        # finite, matches CPU

# Case C: NaN on MetalHLO at batch ≥ 4
jax.jit(lambda x, g, b, w: attn(manual_ln(x, g, b), w))(x, gamma, beta, w_qkv)
```

What's been ruled out for the third bug:

- Every single op tested in isolation at B=4 is correct
  (`debug_attn_inner.py`, `debug_layernorm.py`, `debug_batch.py`).
- Every optimisation pass disabled individually still NaN
  (`norm-fusion`, `ffn-fusion`, `attention-fusion`,
  `producer-consumer-fusion`, `*-canonicalizer`, `final-dce`, …).
- `METALHLO_OPT_LEVEL=O0` still NaN.
- `METALHLO_DEVICE_POLICY=gpu_only` still NaN.
- Two sequential JITs (LN then attention) — works. Combining them
  into one JIT — NaN. Almost certainly a buffer-aliasing or scheduling
  issue inside `IntegratedExecutor` that's triggered by the combined
  op set.

## Files

- `data.py` — tinyshakespeare downloader + char tokenizer + random
  batch sampler.
- `model.py` — Flax NNX nanoGPT (pre-norm blocks, causal MHA, GELU MLP).
- `main.py` — training loop, `--backend {cpu,metalhlo}`. Runs fine on
  CPU. Currently produces `loss = 0.0000` + ~85 s/step on MetalHLO
  because of the third bug.
- `debug_bisect.py` — the original sweep that flagged "manual
  layernorm produces NaN" and the Q@K^T diff.
- `debug_transpose.py` — minimal repro of bug 1 (transpose composition).
  Fixed in commit `5ad3b92`.
- `debug_4dmatmul.py`, `debug_qkv.py` — narrowing of bug 1.
- `debug_layernorm.py`, `debug_var.py` — bisect of jnp.var → NaN.
- `debug_select.py` — minimal repro of bug 2 (scalar-pred select).
  Fixed in commit `5ad3b92`.
- `debug_model_fwd.py` — narrowed batch-size threshold for bug 3 to B≥4.
- `debug_batch.py`, `debug_pipeline.py`, `debug_attn_inner.py`,
  `debug_ln_attn.py` — sequential narrowing of bug 3.

## What needs to happen to ship this benchmark

One of:

1. **Find and fix bug 3.** The composed `attn(ln(x))` JIT case at B≥4
   produces NaN inside `IntegratedExecutor`. Likely needs runtime
   instrumentation rather than another op-level bisect — the
   individual ops all work; only the combined function fails. See
   `debug_ln_attn.py` for the smallest reproducer (4-input function,
   ~30 ops).
2. **Pivot the model to `flax.linen.LayerNorm` +
   `flax.linen.MultiHeadDotProductAttention`** — the Mini-BERT test
   in `flax_metalhlo_e2e.py` proves this exact combination works on
   MetalHLO at similar batch sizes, so it would unblock the benchmark
   without waiting on bug 3.
