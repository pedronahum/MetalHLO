# ResNet18 on CIFAR-10 (Flax NNX) — MetalHLO vs CPU

A reproducible apples-to-apples training benchmark adapted from
[tillahoffmann/jax-mps's `examples/resnet`][upstream], which compares
the jax-mps backend against CPU. This version compares **MetalHLO** —
our PJRT backend for Apple Silicon GPUs — against the JAX CPU backend
on the same workload.

[upstream]: https://github.com/tillahoffmann/jax-mps/tree/main/examples/resnet

## Workload

- **Model**: ResNet18 adapted for 32×32 images (`model.py`, mirrored
  from upstream)
- **Dataset**: CIFAR-10 (`data.py`, downloads to `~/.cache/cifar10/`)
- **Optimizer**: Adam, learning rate 1e-3
- **Batch size**: 256
- **Framework**: Flax NNX

The script measures wall-clock time per training step, discarding the
first half of steps (JIT compile + warmup), and prints the mean of the
second half. Same methodology as upstream.

## Running it

```bash
# Build the PJRT plugin first:
swift build -c release --product PJRTMetalHLO

# Quick smoke test on MetalHLO (30 steps is enough to get a steady-state):
python Examples/Benchmarks/resnet_cifar10/main.py --backend metalhlo --steps 30

# Same on CPU for comparison:
python Examples/Benchmarks/resnet_cifar10/main.py --backend cpu --steps 10

# Full epoch (195 steps):
python Examples/Benchmarks/resnet_cifar10/main.py --backend metalhlo
```

Required Python packages: `jax`, `jaxlib`, `flax`, `optax`, `tqdm`.
The MetalHLO plugin is registered automatically — the script picks up
the dylib from `.build/release/libPJRTMetalHLO.dylib`.

## Results

Measured on **Apple M5 Pro**, JAX 0.10.0 / Flax 0.12.7:

| Backend | Time per step | Speedup |
|---------|---------------|---------|
| MetalHLO (GPU) | 0.275 s | **8.48×** |
| CPU | 2.332 s | 1× |

(MetalHLO time is the mean of three back-to-back runs of 30 steps each
— 0.275 / 0.275 / 0.275 s, zero variance to 3 decimals. Times averaged
over the second half of each run. Loss converges identically on both
backends — 2.367 at init, 1.728 at step 30, bit-equal with CPU.)

For comparison, the upstream `jax-mps` benchmark on an M4 MacBook Air
reports CPU 3.2s vs MPS 0.7s (4.7×). Numbers aren't directly
comparable (different chip, different JAX backend), but MetalHLO now
runs nearly 2× faster than that ratio on the more powerful M5 Pro.

## Files

- `main.py` — Training loop. Mirrors upstream verbatim except for a
  `--backend {cpu,metalhlo}` flag that registers the MetalHLO PJRT
  plugin before the first `jax.devices()` call.
- `model.py` — ResNet18 architecture. Identical to upstream.
- `data.py` — CIFAR-10 loader. Identical to upstream.

## What this benchmark exercised in the backend

This was the first end-to-end test of a real-world strided-conv
network on MetalHLO. It surfaced and drove four changes (each a
measurable step on this benchmark):

1. **`lhsDilation > 1` handling in the generic convolution path** —
   XLA expresses the data-gradient of a strided forward convolution
   as a regular conv with `lhsDilation = forward_stride`, which
   inserts zeros between input samples. MPSGraph's
   `Convolution2DOpDescriptor` has no input-dilation field, so we
   materialize the dilated tensor via `applyBaseDilation`. This
   change made the network compile at all.

2. **`applyBaseDilation` rewritten from scatter-based to
   reshape→pad→reshape→slice**. Fewer ops, no scatter, simpler for
   MPSGraph to fuse.

3. **The deterministic slice+matmul conv-grad fast-paths were
   removed**. They had been added to chase cross-process determinism
   in the cnn_step1 test, but they didn't fully fix it (the test
   tolerance `rtol=1e-1` absorbs the residual drift either way) and
   they replaced each MPSGraph grad call with 9 small matmuls. ~30%
   speedup, ~400 lines deleted, simpler control flow.

4. **Dedicated MPSGraph grad-API fast-paths for both stride-1 and
   strided conv-grads** (`convolution2DWeightsGradient` /
   `convolution2DDataGradient`). The generic conv path was routing
   conv-grads through `convolution2D` with permuted dim_numbers and
   surrounding transposes; the dedicated APIs are MPSGraph's
   preferred entry point and skip the transpose triple. For strided
   forward convs (where XLA emits `lhsDilation` / `rhsDilation =
   forward_stride`), the dedicated APIs accept the forward stride
   directly in their descriptor and avoid the explicit dilation
   materialization entirely.

The combination took the benchmark from ~0.64s/step (with the
deterministic slow path) to ~0.41s/step, and as a side-effect
tightened the Mini-ResNet `train_step_params` correctness from a
generous `2e-2` tolerance (the previous transposed-conv path
introduced drift) to bit-exact `7e-8`.

A second round of profiling-driven work took the same benchmark from
0.41s to **0.275 s/step** (5.6× → 8.48×):

5. **HeterogeneousExecutor caches.** Each PJRT execute was re-running
   the GPU/ANE partitioner over all 3,420 ops (~16 ms wasted) and
   re-extracting the per-partition sub-function + re-hashing
   `module.description` for the compile-cache key (~9 ms more).
   `cachedPartitionPlan` plus a `compiledPartitionsByIndex` cache
   make the first execute pay this cost and subsequent ones skip it.

6. **Zero-copy inputs into IntegratedExecutor.** A 600 MB-per-step
   memcpy bug surfaced via PJRT-level timing: `buildMetalInputs` was
   doing `storage.toData() → device.makeBuffer(bytes:)` on every
   input, every execute. The hot offender was `jit_dynamic_slice` on
   the preloaded CIFAR-10 dataset — 600 MB copied twice per step,
   ~80 ms of pure memcpy. When `BufferStorage.metalBuffer` is already
   present, we now pass that MTLBuffer through directly. The single
   `dynamic_slice` PJRT call went from 82 ms to ~1.9 ms.

7. **BatchNorm training-apply fusion.** JAX emits BN training as a
   ~17-op chain (broadcast(reshape(mean)) → subtract → broadcast(rsqrt
   × gamma) → multiply → add(broadcast(reshape(beta)))) — each step
   round-trips through the full N×H×W×C tensor. A new
   `BatchNormFusionPass` detects this pattern (distinguishing it from
   LayerNorm via the multi-axis reduce that produces the 1D mean) and
   replaces the root op with one `graph.normalize` MPSGraph call.
   20 BN forwards detected on ResNet18 (17 main + 3 stage-transition
   projection BNs). The upstream reduce/mean/variance chain stays in
   place — JAX consumes those same values for the running-stats update
   and for the BN gradient.

The runtime profile harnesses that drove (5)–(7) are committed and
env-gated for follow-up perf work: `METALHLO_PROFILE_GPU=1` (encode
/ CPU dispatch / GPU compute via MPSGraph scheduled+completion
handlers), `METALHLO_PROFILE_HET=1` (heterogeneous executor stage
times), and `METALHLO_PROFILE_PJRT=1` (per-PJRT-call timing at the
C-API boundary).
