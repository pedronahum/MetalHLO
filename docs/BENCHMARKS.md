# MetalHLO Benchmarks

This document holds the full benchmark data and the end-to-end performance
investigations. For a high-level summary see the [README](../README.md#performance).

There are two measurement campaigns:

1. **Multi-backend comparison** on Apple M1 (8 GB) — MPSGraph vs Metal O2 vs Metal O3
   vs heterogeneous GPU+ANE+CPU, across MetalHLO's own backends.
2. **MLX comparison** on Apple M5 Pro (48 GB) — MetalHLO's codegen path vs
   [MLX](https://github.com/ml-explore/mlx), the reference hand-tuned numerical library.

---

## Multi-Backend Comparison (Apple M1)

All results measured on **Apple M1 (8 GB)**, macOS 15.6, release build, quick mode
(3 warmup, 10 measurements). Times are mean in milliseconds. **Bold** indicates the
fastest backend for each benchmark.

### Matrix Operations

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| MAT-DOT-001 | GEMM 128x128 | 0.30 | **0.17** | 0.24 | 0.25 | 1.72x (O2) |
| MAT-DOT-002 | GEMM 512x512 | 1.03 | **0.77** | 1.71 | 0.85 | 1.34x (O2) |
| MAT-DOT-003 | GEMM 1024x1024 | **4.17** | 5.10 | 5.32 | 4.53 | MPSGraph best |
| MAT-DOT-004 | GEMM 2048x2048 | **19.55** | 23.00 | 22.63 | 22.17 | MPSGraph best |
| MAT-DOT-005 | GEMM 4096x4096 | **122.7** | 226.1 | 227.6 | 234.4 | MPSGraph best |
| MAT-DOT-006 | Transformer 32x4096x768 | 2.36 | 1.22 | 1.17 | **1.10** | 2.14x (ANE) |
| MAT-DOT-007 | MLP 128x768x3072 | 3.11 | 1.90 | 2.06 | **1.63** | 1.91x (ANE) |
| MAT-DOT-008 | Matvec 1x4096x4096 | **3.93** | 10.91 | 11.68 | 10.94 | MPSGraph best |
| MAT-BATCH-001 | Batched 8x512x512 | 6.10 | 5.41 | **4.69** | 5.58 | 1.30x (O3) |
| MAT-BATCH-002 | Batched 4x1024x1024 | 4.59 | **2.85** | 3.03 | 2.99 | 1.61x (O2) |
| MAT-BATCH-003 | Attention heads | 1.61 | 1.12 | **0.67** | 1.28 | 2.42x (O3) |
| MAT-BATCH-004 | Multi-head attention | 0.81 | 0.38 | 0.38 | **0.36** | 2.29x (ANE) |
| MAT-TR-001 | Transpose 1024x1024 | 1.45 | 0.84 | **0.50** | 1.27 | 2.87x (O3) |
| MAT-TR-002 | Transpose 3D 32x128x64 | 0.49 | 0.65 | **0.25** | 0.56 | 1.99x (O3) |
| MAT-RSH-001 | Reshape flatten 1024x1024 | 1.34 | 0.53 | **0.48** | 0.57 | 2.80x (O3) |
| MAT-RSH-002 | Reshape batch 32x64x128 | 0.48 | **0.41** | 0.46 | 0.42 | 1.16x (O2) |

**Takeaway:** MPSGraph wins on large GEMMs (≥1024x1024) and matvec where Apple's tuned
kernels dominate. Metal O3 wins on batched operations, transpose, and reshape (1.3-2.9x
faster). O2 wins on small-to-mid GEMMs. ANE excels on transformer-shaped and MLP matmuls.

### Element-wise Arithmetic

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| ARITH-B-001 | Add 1024x1024 | 2.52 | 0.96 | **0.75** | 0.75 | 3.35x (ANE) |
| ARITH-B-002 | Add 4096x4096 | 12.20 | 10.30 | 16.43 | **8.75** | 1.40x (ANE) |
| ARITH-B-003 | Add 8192x8192 | **57.7** | 169.1 | 138.5 | 148.1 | MPSGraph best |
| ARITH-B-004 | Mul 1024x1024 | 2.66 | **0.79** | 0.83 | 0.88 | 3.38x (O2) |
| ARITH-B-005 | Mul 4096x4096 | 11.91 | **10.12** | 10.43 | 13.97 | 1.18x (O2) |
| ARITH-B-006 | Div 1024x1024 | 2.16 | 0.97 | **0.80** | 1.15 | 2.69x (O3) |
| ARITH-B-007 | Pow 1024x1024 | 2.52 | **0.97** | 0.98 | 1.03 | 2.59x (O2) |
| ARITH-B-008 | Max 4096x4096 | 11.85 | 12.46 | 9.42 | **8.32** | 1.42x (ANE) |
| ARITH-U-001 | Exp 1024x1024 | 1.23 | 0.61 | **0.57** | 0.85 | 2.14x (O3) |
| ARITH-U-002 | Log 4096x4096 | 8.07 | 5.01 | **4.90** | 5.08 | 1.65x (O3) |
| ARITH-U-003 | Tanh 1024x1024 | 1.54 | **0.52** | 0.52 | 0.81 | 2.96x (O3) |
| ARITH-U-004 | Sqrt 4096x4096 | 8.02 | **4.61** | 4.79 | 5.30 | 1.74x (O2) |
| ARITH-U-005 | Rsqrt 4096x4096 | 8.71 | 5.65 | 5.74 | **5.57** | 1.56x (ANE) |
| ARITH-U-006 | Sigmoid 1024x1024 | 1.34 | 0.70 | **0.70** | 0.79 | 1.93x (O3) |
| ARITH-BC-001 | Add row broadcast | 1.35 | 1.34 | **0.93** | 1.10 | 1.45x (O3) |
| ARITH-BC-002 | Add scalar broadcast | 1.29 | 1.00 | 0.99 | **0.81** | 1.58x (ANE) |
| ARITH-BC-003 | Mul last-dim broadcast | 0.48 | **0.39** | 0.44 | 0.54 | 1.23x (O2) |

**Takeaway:** Metal backends are **1.2-3.4x faster** than MPSGraph on element-wise
operations. The advantage is largest on 1024x1024 tensors (2.1-3.4x); on 4096x4096
tensors MPSGraph has improved significantly (macOS 15.6), narrowing the gap to 1.2-1.7x.

### Reduction Operations

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| RED-001 | Global sum 1024x1024 | 0.91 | 0.96 | 0.89 | **0.86** | 1.06x (ANE) |
| RED-002 | Row-wise sum 1024x1024 | 1.03 | **0.87** | 1.09 | 0.95 | 1.18x (O2) |
| RED-003 | Column-wise sum 1024x1024 | 0.90 | **0.82** | 0.83 | 0.84 | 1.10x (O2) |
| RED-004 | Row-wise max 4096x4096 | **4.21** | 6.25 | 5.91 | 6.17 | MPSGraph best |
| RED-005 | LayerNorm reduction 32x128x768 | 2.72 | 1.29 | **0.95** | 1.09 | 2.86x (O3) |
| RED-006 | Attention reduction 32x12x512x512 | **22.49** | 47.37 | 46.81 | 37.49 | MPSGraph best |

**Takeaway:** Metal backends win on 1024x1024 reductions (1.1-1.2x). O3 excels on
LayerNorm reduction (2.9x). On larger reductions MPSGraph wins (macOS 15.6 improvements).

### Convolution

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| CONV-001 | ResNet first layer | **1.92** | 1.91 | 2.24 | 2.78 | ~1.00x |
| CONV-002 | ResNet stage2 3x3 | **1.16** | 3.36 | 3.09 | 1.16 | MPSGraph best |
| CONV-003 | ResNet stage3 3x3 | **1.06** | 3.64 | 3.15 | 1.36 | MPSGraph best |
| CONV-004 | ResNet stage4 3x3 | **1.73** | 3.45 | 2.77 | 2.01 | MPSGraph best |
| CONV-005 | Batched conv | **11.79** | 34.94 | 34.42 | 20.22 | MPSGraph best |
| CONV-006 | 1x1 pointwise | 0.83 | 0.71 | **0.68** | 1.75 | 1.22x (O3) |
| CONV-007 | Depthwise-like | 1.61 | 4.88 | 5.24 | **1.54** | 1.05x (ANE) |

**Takeaway:** MPSGraph dominates convolutions thanks to Apple's highly optimized
`MPSCNNConvolution` kernels. Metal O3 wins on 1x1 pointwise; ANE occasionally beats
MPSGraph on depthwise patterns.

### Normalization

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| NORM-BN-001 | ResNet BN | 0.42 | **0.24** | 0.28 | 0.39 | 1.76x (O2) |
| NORM-BN-002 | Batched ResNet BN | 8.06 | 3.83 | **2.43** | FAIL | 3.32x (O3) |
| NORM-BN-003 | Mid-layer BN | 0.35 | **0.18** | 0.21 | 0.19 | 1.90x (O2) |
| NORM-BN-004 | Late-layer BN | 0.29 | **0.17** | 0.38 | 0.19 | 1.70x (O2) |
| NORM-LN-001 | BERT-base LayerNorm | 0.57 | **0.52** | 0.56 | 0.59 | 1.09x (O2) |
| NORM-LN-002 | BERT-base batched LN | **4.42** | 6.17 | 6.87 | 6.46 | MPSGraph best |
| NORM-LN-003 | BERT-large single LN | **1.36** | 1.77 | 2.17 | 1.63 | MPSGraph best |
| NORM-LN-004 | Long sequence LN | **15.67** | 20.90 | 21.10 | 23.02 | MPSGraph best |

**Takeaway:** Metal O2 excels at batch normalization (1.7-1.9x). O3 wins on large batched
BN (3.3x). MPSGraph wins on layer normalization.

### Transformer Components

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| XFMR-INF-001 | Self-attention seq=128 | 2.76 | 2.51 | **2.19** | 2.41 | 1.26x (O3) |
| XFMR-INF-002 | Self-attention seq=512 | **7.61** | 7.85 | 8.28 | 7.82 | MPSGraph best |
| XFMR-INF-003 | Self-attention BS=8 seq=128 | **8.80** | 9.45 | 9.43 | 9.22 | MPSGraph best |
| XFMR-INF-004 | Transformer FFN BS=8 | **11.27** | 18.24 | 16.71 | 16.02 | MPSGraph best |
| XFMR-INF-005 | Softmax 8x12x128x128 | 2.53 | 2.34 | 2.29 | **2.29** | 1.11x (ANE) |
| XFMR-INF-006 | Encoder block BS=1 seq=128 | 7.71 | **7.46** | 8.48 | 7.84 | 1.03x (O2) |

**Takeaway:** MPSGraph wins on larger sequence lengths, FFN, and batched attention. O3
excels on self-attention (1.3x). ANE is competitive on softmax.

### MLP Inference

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| MLP-INF-001 | 784->256->10 BS=1 | 0.64 | **0.45** | 0.45 | 0.80 | 1.43x (O2) |
| MLP-INF-002 | 784->256->10 BS=32 | **0.41** | 0.43 | 0.44 | 0.60 | MPSGraph best |
| MLP-INF-003 | 784->256->10 BS=128 | 0.60 | 0.72 | 0.47 | **0.45** | 1.35x (ANE) |
| MLP-INF-004 | Deep MLP 4-layer BS=32 | 1.05 | 0.91 | 0.80 | **0.75** | 1.40x (ANE) |
| MLP-INF-005 | FFN 768->3072->768 BS=32 | 4.35 | 1.77 | **1.58** | 1.84 | 2.75x (O3) |

**Takeaway:** Metal backends are 1.4-2.8x faster than MPSGraph on MLPs. O3 wins the large
FFN (2.8x); ANE excels on deep and batched MLPs.

### Training

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| TRAIN-001 | MLP fwd+bwd BS=32 | **0.69** | 0.91 | 0.90 | 1.00 | ~1.00x |
| TRAIN-003 | Attention fwd+bwd BS=8 | **7.25** | 10.29 | 10.82 | 11.12 | MPSGraph best |

**Takeaway:** MPSGraph leads on training workloads where its backward-pass graph
optimization provides an advantage.

### Backend Win Summary

Overall wins across all 67 passing benchmarks:

| Backend | Wins | Best For |
|---------|------|----------|
| **MPSGraph** | 20 | Large GEMMs, convolutions, layer norm, large reductions, training |
| **Metal O2** | 18 | Batch norm, small matmuls, element-wise ops |
| **Metal O3** | 16 | Transpose, reshape, batched ops, softmax, FFN fusion |
| **GPU+ANE** | 13 | Element-wise, MLP inference, transformer-shaped matmuls |

### When to Use Each Backend

| Use Case | Recommended Backend | Why |
|----------|-------------------|-----|
| Large matrix multiply (≥1024x1024) | MPSGraph (default) | Apple's tuned `MPSMatrixMultiplication` |
| Convolution-heavy models (CNNs) | MPSGraph (default) | `MPSCNNConvolution` is highly optimized |
| Training (forward + backward) | MPSGraph (default) | Graph-level backward pass optimization |
| Element-wise heavy workloads | Metal O2/O3 | 1.2-3.4x faster than MPSGraph |
| Batch normalization | Metal O2 | 1.7-1.9x faster custom kernels |
| Transpose / reshape / batched ops | Metal O3 | Pattern fusion + scheduling (2-3x faster) |
| MLP / FFN inference | Metal O3 or ANE | O3 wins large FFN (2.8x), ANE wins deep MLPs |
| Debugging/development | MPSGraph (default) | Broadest compatibility, no compilation |

---

## MLX Comparison (Apple M5 Pro)

Direct comparison against [MLX](https://github.com/ml-explore/mlx) on Apple M5 Pro
(48 GB), macOS 26.4.1, Xcode 26.4. MLX is the reference for "what an end-to-end-tuned
numerical kernel library can achieve on Apple Silicon," so it's the right yardstick for
the codegen path.

Measurements use the standalone `mlx-comparison` runner in **quick mode** (3 warmup, 10
measurements), `METALHLO_MATMUL_TF32=1`. **Speedup > 1.0x means MetalHLO is faster than
MLX**; speedup < 1.0x means MLX is faster.

### Per-Category Geomean (60 benchmarks)

The two columns show -O0 (no optimizer passes — kernel-level performance only) and -O3
(full pattern fusion: FFN, attention, LayerNorm fold into single Metal kernels).

| Category | Benchmarks | -O0 | **-O3** | -O3 wins |
|---|---|---|---|---|
| **model_mlp** | 5 | 1.28x | **1.81x** | **5 / 5** |
| **normalization** | 4 | 0.60x | **1.09x** | 3 / 4 |
| **matrix** | 16 | 0.76x | 0.72x | 4 / 16 |
| **model_transformer** | 5 | 0.44x | 0.69x | 1 / 5 |
| **arithmetic** | 17 | 0.57x | 0.70x | 2 / 17 |
| **convolution** | 7 | 0.44x | 0.43x | 1 / 7 |
| **reduction** | 6 | — | ~0.70x | 1 / 6 |

(Reduction was re-measured after the MLX-mirroring float4 row-reduction kernel
landed: geomean is now ~0.67–0.85x — quick-mode noise is large on sub-ms
kernels — with one win, RED-004. The remaining gaps are the **global**
reduction RED-001 (0.27x — runs in a single threadgroup) and the **column**
reduction RED-003 (0.48x — strided/uncoalesced reads), not the axis/softmax
reductions, which are now competitive.)

Three categories see large -O3 jumps:
- **model_mlp** sweeps 5/5 against MLX. The FFN detector recognizes the canonical SiLU
  (`multiply(x, logistic(x))`) and ReLU (`maximum(x, 0)`) lowerings, so each
  `matmul → activation → matmul` block fuses into one `fused_ffn` Metal kernel.
- **normalization** crosses MLX parity. The LayerNorm detector recognizes the expanded
  form `add(multiply(multiply(subtract(x, mean), rsqrt(...)), gamma), beta)` and folds it
  into one `fused_layer_norm`.
- **model_transformer** improves but doesn't yet beat MLX on attention. The detector fuses
  Q@K + softmax + @V into one `fused_scaled_dot_product_attention`; MLX's
  `scaled_dot_product_attention` is still ~30% faster on the 4/5 sequences measured.

### Headline Numbers

All times in milliseconds, measured at -O3 with `METALHLO_MATMUL_TF32=1` on M5 Pro.

| ID | Description | MetalHLO | MLX | Speedup | Notes |
|---|---|---|---|---|---|
| MAT-DOT-005 | GEMM 4096² | 8.24 | 7.47 | **0.91x** | Matrix coprocessor; gap is ~1ms TF32 input-convert overhead |
| MAT-DOT-002 | GEMM 512² | 0.53 | 0.54 | **1.02x** | Parity |
| MLP-INF-001 | MLP 784→256→10 BS=1 (3-layer ReLU) | 0.22 | 0.27 | **1.22x** | FFN fusion + chain ReLU detection |
| MLP-INF-005 | FFN 768→3072→768 BS=32 (SiLU) | 0.86 | 3.97 | **4.63x** | FFN fusion |
| ATTN-002 | Self-attention BS=8 H=12 S=128 D=64 | 0.65 | 0.45 | 0.69x | Attention fusion fires; MLX's primitive ~30% faster |
| NORM-LN-002 | LayerNorm 32×128×768 (BERT-base batched) | 1.42 | 0.66 | 0.46x | LayerNorm fusion fires; MLX still ahead on this shape |
| RED-006 | Sum-reduce axis-3 32×12×512² | 1.91 | 1.54 | 0.81x | MLX-mirroring float4 row-reduction kernel (was 0.27x before that kernel) |
| RED-001 | Global sum 1024² | 1.00 | 0.27 | 0.27x | Single-threadgroup bottleneck (worst remaining reduction) |

**Takeaway.** Three classes of speedup are active in -O3:
1. **Kernel-level**: MAT-DOT-005 hits 0.91x of MLX via the MetalPerformancePrimitives
   matrix coprocessor primitive. The remaining ~1ms is the fp32→fp16 input-convert
   overhead the TF32 transform pays.
2. **Pattern fusion**: FFN, attention, and LayerNorm patterns expressed as expanded
   primitive ops collapse into single fused Metal kernels. MLP-INF-005 hits 4.63x; all 5
   model_mlp benchmarks beat MLX.
3. **Reductions**: the common axis/row reductions (RED-006, softmax-shaped) are now
   competitive (0.81x) via the MLX-mirroring float4 kernel; the remaining gaps are the
   global (RED-001, 0.27x — single-threadgroup) and column (RED-003, 0.48x — strided)
   reductions. **Convolution**: still bounded by per-kernel performance with no fused-pattern
   path, so MLX's hand-tuned kernels remain ahead.

### Reproducing These Numbers

Requires macOS 26+ and an Apple9-class GPU (M3, M3 Pro/Max/Ultra, M4, M5, M5 Pro/Max/Ultra)
for the MPP path; on older OS or GPUs the build still works and the runtime falls back to
the simdgroup_matrix kernel automatically.

```bash
# Build the comparison runner. Must be xcodebuild, not `swift build` —
# MLX bundles a Metal library that swiftpm doesn't compile.
xcodebuild build \
  -scheme mlx-comparison \
  -configuration Release \
  -destination 'platform=OS X' \
  -derivedDataPath .build/xcode

BIN=.build/xcode/Build/Products/Release/mlx-comparison
RESULTS=results/mlx_comparison_m5pro
mkdir -p "$RESULTS"

# Run all 7 categories at -O3, save JSON + Markdown per category.
for cat in matrix arithmetic reduction convolution \
           normalization model_mlp model_transformer; do
  METALHLO_MATMUL_TF32=1 "$BIN" --quick -O3 -c "$cat" \
    -o "$RESULTS/$cat.json"
done

# Single-benchmark spot check (the headline matmul):
METALHLO_MATMUL_TF32=1 "$BIN" --quick -O3 -f MAT-DOT-005

# Show which optimizer passes fire on a given benchmark:
METALHLO_DEBUG_PASSES=1 METALHLO_MATMUL_TF32=1 "$BIN" --quick -O3 -f MLP-INF-005

# Force the simdgroup_matrix fallback path:
METALHLO_MATMUL_TF32=1 METALHLO_DISABLE_MPP=1 "$BIN" --quick -O3 -f MAT-DOT-005
```

**Environment variables that affect this comparison:**

| Variable | Default | Effect |
|---|---|---|
| `METALHLO_MATMUL_TF32` | `0` | When `1`, wraps fp32 dot/dot_general with `convert(fp32→fp16)` so the matmul runs at fp16. The MPP kernel uses `half × half → float` natively. Required to hit MLX-class throughput on large GEMMs. |
| `METALHLO_DISABLE_MPP` | `0` | When `1`, disables the MetalPerformancePrimitives matmul path even on capable hardware — falls back to the simdgroup_matrix kernel. Kill switch for diagnosing kernel-compile/correctness regressions. |
| `METALHLO_FORCE_MPSGRAPH` | `0` | When `1`, routes every op through MPSGraph instead of codegen. Bypasses the optimizer; useful as a sanity check. |
| `METALHLO_DEBUG_PASSES` | `0` | When `1`, the optimizer prints one line per pass (`[*] pass-name ops: N -> M`, asterisk = changed) to stderr. |

### Caveats

- **Hardware-specific.** All numbers are M5 Pro / macOS 26.4.1. The MPP path requires
  Apple9 GPU family and Metal language 4.0; on M1/M2 or pre-macOS-26 the runtime gate
  silently selects the simdgroup_matrix fallback, with results similar to the M1 tables above.
- **Run-to-run variance.** Quick mode introduces ±10–15% noise on benchmarks under 1ms.
  The headline numbers (MAT-DOT-005, MLP-INF-005) are stable; small-shape numbers fluctuate.
- **MLX is the ceiling, not the universal target.** MLX is a hand-tuned numerical library;
  matching it on every shape isn't the goal. MetalHLO's reason to exist is the optimizer +
  heterogeneous fusion that MLX doesn't have.

---

## End-to-End Training Gap (nanoGPT) — Investigation Status

The microbenchmarks above are forward-only (inference), where pattern fusion is free to
fold FFN / attention / LayerNorm into single kernels. The harder, ongoing target is a full
**training step** — a 6-layer / 384-dim / 6-head nanoGPT (batch 16, seq 256, ~10.8 M
params, char-level tinyshakespeare; `Examples/Benchmarks/nanogpt`) — where the reverse-mode
autograd graph changes what fusion can do.

**Where we stand (M5 Pro):** **74.4 ms/step vs MLX 59.5 ms (1.25×)**, down from 82.5 ms.
The loss matches JAX CPU exactly at every step throughout.

**Step composition.** GPU-busy is ~60.8 ms; the rest is host overhead (the per-step output
handoff copy, command-buffer encode, PJRT / `.item()` sync). The pivotal fact: **GPU-busy
alone already ≈ MLX's *entire* 59.5 ms step.** So the tractable headroom was the host
overhead, not the GPU-side glue.

**Landed (all loss-neutral):**
- **Reshape-as-view** — a reshape of a contiguous source (a kernel output or input) aliases
  that buffer instead of running a copy kernel, with the memory planner keeping the source
  live across the view's readers. Eliminated ~95 kernels/step.
- **Parallel output handoff** — the 162-output (~130 MB) copy out of the reused intermediate
  slab now runs concurrently instead of serially.
- **Pipelined command buffers** — the ~900-kernel encode is split across command buffers
  committed incrementally, so the GPU runs chunk *k* while the CPU encodes chunk *k+1*.

**Investigated, found bounded by reverse-mode autodiff** — three glue levers each profiled
to a dead end:
- **Rematerialization** — this MLP is ReLU (no expensive forward intermediate like a GELU
  `tanh(x³)` worth recomputing), and roughly half the standalone glue is *terminal gradient
  outputs* (return values, unfusable by anything).
- **Transpose-into-matmul** — the simple `transA/transB` cases are already absorbed natively
  by the matmul path; the survivors are 4-D attention head-permutations that would need a
  strided-operand matmul kernel (unverified hardware surface, ~3 ms ceiling).
- **Chain / reduce fusion** — the valuable same-shape chain merges are already done by
  producer-consumer fusion; the remaining chain boundaries are either *multi-use* (a forward
  value also consumed by the backward pass) or broadcast-shape-changes of a tiny reduced
  tensor (negligible traffic saved).

The common root is structural: **reverse-mode AD materializes forward activations for the
backward pass, making them multi-use** — which both breaks elementwise chains at every
branch and makes forward pattern fusion unsafe on the training graph. With GPU-busy already
at MLX's step time and matmul (~40% of GPU) at the TF32 / matrix-coprocessor limit, the
residual gap is systemic rather than a single missing optimization. nanoGPT runs at the
default optimization level (the gpu-only codegen path), not -O3.

**Reproduce / profile:**

```bash
# steady-state ms/step + final loss (compare against the JAX CPU baseline)
python Examples/Benchmarks/nanogpt/main.py --backend metalhlo --steps 30 --skip-inference
python Examples/Benchmarks/nanogpt/main.py --backend cpu       --steps 30 --skip-inference

METALHLO_PROFILE_GPU=1    ...   # GPU-busy vs wall-clock per step
METALHLO_PROFILE_PER_OP=1 ...   # per-kernel-type GPU share
METALHLO_PIPELINE_CHUNK=0 ...   # disable encode pipelining (single command buffer)
```

---

## Heterogeneous Execution (GPT-2 Validation)

The heterogeneous GPU+ANE+CPU pipeline was validated on GPT-2 (124M parameters):

| Metric | Result |
|--------|--------|
| **Logit projection speedup** | 1.91x (fused GPU+MPS+CPU vs single-unit) |
| **Regressions** | Zero — compound gate rejects all unprofitable ops |
| **Ops partitioned (seq=512)** | 1 of 147 (logit projection only) |
| **Cross-architecture** | ViT-B/16: 0 partitioned at all batch sizes (correct) |

The compound gate is **batch-invariant**: the same ops are selected regardless of batch
size (1 through 8). At batch ≥ 16 on 8GB devices, memory pressure causes regressions — the
gate should be extended with an upper memory bound for large-batch safety.

---

## Running Benchmarks

```bash
# Build in release mode for accurate measurements
swift build -c release --product benchmark-runner

# Multi-backend comparison (recommended)
.build/release/benchmark-runner --compare -q -c matrix             # Matrix operations
.build/release/benchmark-runner --compare -q -c arithmetic         # Element-wise ops
.build/release/benchmark-runner --compare -q -c model_transformer  # Transformer
.build/release/benchmark-runner --compare -q                       # All categories

# Single-backend benchmarks
.build/release/benchmark-runner --category matrix
.build/release/benchmark-runner --all

# MLX comparison (requires MLX) — see "Reproducing These Numbers" above for the
# xcodebuild path, which is required because MLX bundles its own Metal library.

# Heterogeneous fusion benchmarks (GPU+MPS+CPU partitioning)
swift build -c release --product HeterogeneousFusionBenchmark
.build/release/HeterogeneousFusionBenchmark

# GPT-2 end-to-end validation (profitability guard + crossover analysis)
swift build -c release --product GPT2EndToEnd
.build/release/GPT2EndToEnd
```

### Benchmark Categories

| Category | Benchmarks | Description |
|----------|------------|-------------|
| **Matrix Operations** | MAT-DOT-*, MAT-BATCH-*, MAT-TR-*, MAT-RSH-* | GEMM, batched GEMM, transpose, reshape |
| **Arithmetic** | ARITH-B-*, ARITH-BC-*, ARITH-U-* | Binary, broadcast, and unary operations |
| **Reduction** | RED-001 to RED-009 | Sum, max, mean, pooling operations |
| **Convolution** | CONV-001 to CONV-007 | Standard conv2d patterns (ResNet, VGG) |
| **Normalization** | NORM-BN-*, NORM-LN-* | Batch norm, layer norm |
| **Control Flow** | CF-001 to CF-005 | While loops, conditionals |
| **Indexing** | IDX-001 to IDX-007 | Slice, gather, scatter, pad |
| **Model Inference** | MLP-INF-*, CNN-INF-*, XFMR-INF-* | MLP, CNN, Transformer components |
| **Training** | TRAIN-001 to TRAIN-003 | Forward + backward pass benchmarks |
| **Compiler Analysis** | COMP-001 to COMP-005 | Compilation time for various program sizes |
| **Fusion Analysis** | FUSION-001 to FUSION-004 | Fused vs naive execution comparison |
| **Memory** | MEM-001 to MEM-003 | Peak allocation, buffer reuse |
| **Power Efficiency** | PWR-001 to PWR-003 | Throughput per watt estimates |

### Framework Features

The benchmark framework provides timing statistics (mean, std dev, min, max, p95, p99),
GPU-synchronized timing via Metal command buffer completion, configurable warmup, seeded
random data, and console/JSON/CSV output. It can be driven programmatically:

```swift
import MetalHLOBenchmarks

let config = BenchmarkConfig(warmupIterations: 10, measurementIterations: 50)
let runner = try BenchmarkRunner(config: config)
let results = try runner.run(OperationBenchmarks.matrixBenchmarks())
for result in results {
    print("\(result.id): \(result.timing.mean * 1000)ms ± \(result.timing.stdDev * 1000)ms")
}
```
