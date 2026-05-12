# Karpathy's atomic GPT — pure Python vs JAX (CPU / MetalHLO)

A three-way comparison of the **same model, same data, same Adam
optimizer** running on:

1. **Pure Python** with hand-rolled scalar autograd, dependency-free —
   verbatim from [@karpathy's gist][gist].
2. **JAX on CPU** — the same math expressed as `jax.numpy` arrays and
   `jax.jit`'d.
3. **JAX on MetalHLO** — the same JAX code, routed through our PJRT
   plugin onto the Apple Silicon GPU.

[gist]: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

## Workload

- **Model**: 1-layer GPT, `n_embd=16`, `n_head=4`, `block_size=16`,
  `vocab_size=27` (26 lowercase letters + BOS), **4,192 parameters total**.
  RMSNorm (no LayerNorm), ReLU (no GELU), no biases.
- **Dataset**: 32,033 lowercase names from
  [karpathy/makemore][makemore-names].
- **Training**: 1,000 steps, one document per step (no batching, exactly
  as the gist does it), Adam with linear LR decay (`lr_t = 0.01 *
  (1 - step/num_steps)`, `β1=0.85`, `β2=0.99`, `ε=1e-8`).
- **Loss**: average cross-entropy per valid (non-padding) position.

[makemore-names]: https://github.com/karpathy/makemore/blob/master/names.txt

The JAX version is the same model: it pads each document to `block_size`
and runs the whole sequence through GPT in **one fixed-shape forward
pass** with a causal mask, instead of the gist's growing-KV-cache loop.
That's the same math at every valid position — the loss is masked over
padding and ignored at training time. **Init weights are drawn from
Python's `random.gauss` with the same seed**, so the parameter tensors
start out bit-equal across the pure-Python and JAX runs.

## Results

Measured on **Apple M5 Pro**, JAX 0.10.0 / Flax 0.12.7, fp32. Timings
are end-to-end wall clock for the training loop (excluding model
inference at the end); the mean step time discards the first 10% as
JIT-compile warmup.

| Backend     | 1000-step total | Mean step  | Speedup vs Pure Python | Final loss |
|-------------|-----------------|------------|------------------------|------------|
| Pure Python | 51.04 s         | 51.0 ms    | 1×                     | 2.6497     |
| JAX CPU     | 0.36 s          | 0.08 ms    | **141×**               | 2.6497     |
| JAX MetalHLO| 13.74 s         | 10.1 ms    | **3.7×**               | 2.6397     |

**Pure Python → JAX CPU loss matches exactly** (2.6497 at step 1000
and at every printed checkpoint), confirming the JAX port is faithful
to the gist's math — same Adam updates, same per-position
cross-entropy, same init. JAX-MetalHLO drifts slightly (2.6397) from
floating-point reduction ordering differences on the GPU; both reach
the same loss neighbourhood.

### Why MetalHLO is *slower* than JAX CPU here

This is the headline lesson of the benchmark. The model is microscopic
— each training step does roughly 10 thousand multiply-adds of real
work. JAX's CPU JIT compiles that to inline native code with no
dispatch overhead at all; the whole step amortises to ~80 µs. The
MetalHLO path, on the other hand, pays a fixed cost per training step:
PJRT execute callback, MPSGraph command-buffer encode, Metal kernel
launch, GPU→CPU `.item()` sync. That fixed cost is in the 5–10 ms
range — orders of magnitude larger than the compute time on a model
this small.

This is the opposite character from the
[ResNet18/CIFAR-10 benchmark](../resnet_cifar10) next door, where the
compute per step (~430 GFLOPs) is large enough that the same GPU
dispatch overhead disappears into the noise and MetalHLO beats JAX
CPU 8.5×.

In other words: **GPU acceleration is workload-dependent.** Tiny
models with frequent updates are dominated by per-step overhead, where
a tight CPU JIT wins. Compute-bound networks let the GPU stretch its
legs. Both benchmarks are useful evidence about where the backend
helps and where it doesn't.

### Generated samples (after 200 steps)

After 200 training steps, the model has learned roughly what English
names look like (vowel placement, name-like endings). Each backend
produces similar-quality samples — the per-step LR / fp drift shows
up in the specific tokens chosen, not in the overall name-ness:

```
JAX CPU:    jamin, antri, jadin, auma, namada, arlar, siba, …
MetalHLO:   jaiss, aloyn, jaaon, auma, makaa,  jarie, ussal, …
```

(Inference uses Python's `random.choices` with the same seed across
runs; sampled tokens differ because the post-training logit values are
not bit-equal across backends.)

## Running it

```bash
# 1) Pure Python — no dependencies beyond the Python standard library.
#    System python3.11 is fine. ~51s on an M5 Pro.
python3 Examples/Benchmarks/karpathy_gpt/pure_python.py --steps 1000

# 2) JAX CPU — uses the venv at .venv/.  ~0.36s.
.venv/bin/python Examples/Benchmarks/karpathy_gpt/jax_gpt.py \
    --backend cpu --steps 1000

# 3) JAX on MetalHLO — needs the PJRT plugin built (`swift build -c release`)
#    and the venv at .venv/.  ~13.7s.
METALHLO_PYTHON=$PWD/.venv/bin/python \
    .venv/bin/python Examples/Benchmarks/karpathy_gpt/jax_gpt.py \
    --backend metalhlo --steps 1000
```

Common flags on both Python files:

- `--steps N` — number of training steps (default 1000).
- `--skip-inference` — skip the 20-sample generation loop after
  training (training-only timing).
- `--print-every K` — progress-line update interval.

The `input.txt` dataset (≈198 KB) is downloaded automatically on first
run and cached next to the script.

## Files

- `pure_python.py` — Karpathy's gist code, lightly wrapped with
  `argparse` + wall-clock timing. The training-loop body, autograd
  `Value`, and `gpt(...)` model function are byte-for-byte the
  original.
- `jax_gpt.py` — JAX port. Same hyperparameters, same hand-rolled Adam
  state, same per-position cross-entropy loss; just expressed as
  `jax.numpy` ops with a JIT-compiled training step. `--backend
  {cpu,metalhlo}` flag mirrors the ResNet18 benchmark layout.
