# vivsim Examples: MetalHLO Backend Test Findings

**Date:** 2026-04-03
**Environment:** macOS Darwin 24.6.0, Apple M1 8GB, JAX 0.7.0, MetalHLO (release build)

## Summary

| Example | CPU | MetalHLO | Issue |
|---------|-----|----------|-------|
| lid_driven_cavity | OK | OK (28.5s vs 1.7s CPU) | Works, float32 diffs ~1e-6 |
| poiseuille_channel | OK | CRASH | Apple MPSGraph internal reshape bug |
| flow_pass_cylinder | OK | CRASH | Apple MPSGraph internal reshape bug |
| flow_through_text | OK | CRASH | Apple MPSGraph internal reshape bug |
| vortex_induced_vibration | OK | CRASH | Parser undefined value (fixed, needs retest with MPSGraph fix) |

## Issues Found and Fixed

### 1. Pipe Deadlock in Bytecode Conversion (FIXED)

**Symptom:** JIT compilation of any non-trivial function hangs indefinitely.

**Root Cause:** Classic pipe deadlock in `PJRTExecutable.swift:convertBytecodeToText()`. Synchronous stdin write + deferred stdout read. If subprocess output exceeds pipe buffer, both processes block.

**Fix:** Write stdin on background `DispatchQueue`, read stdout/stderr before `waitUntilExit()`.

**File:** `Sources/PJRTMetalHLO/PJRTExecutable.swift`

### 2. Constant Pre-compilation (FIXED)

**Symptom:** `Undefined value: %cst_43` on complex HLO graphs.

**Root Cause:** Constants defined after their first use in MLIR text. The compiler iterates operations sequentially and assumes definitions precede uses.

**Fix:** Two-pass compilation — pre-compile all constants before other operations.

**File:** `Sources/MetalHLOCore/Compiler/MPSGraphCompiler.swift`

### 3. MLIR Call Inliner (IMPLEMENTED, DISABLED)

**Symptom:** All `@jax.jit` functions in JAX 0.7 produce `call @` wrappers, forcing MPSGraph path.

**Implementation:** Text-level inliner detects trivial wrapper pattern (`@main` calls `@private_func`, returns results) and replaces with just the private function renamed to `@main`. Skips functions with nested calls.

**Status:** Disabled because the O2 Metal kernel path doesn't handle multi-output functions correctly (PJRT output memory kind validation fails). The MPSGraph path handles calls natively.

**File:** `Sources/PJRTMetalHLO/PJRTExecutable.swift` (function `inlineSimpleCallWrapper`)

### 4. Apple MPSGraph Internal Reshape Bug (OPEN - Apple Framework Bug)

**Symptom:** 3 examples crash during `graph.compile()` with:
```
'mps.reshape' op the result shape is not compatible with the input shape
(tensor<2xsi32>, tensor<3xsi32>) -> tensor<9x1x1xsi32>
```

**Root Cause:** This is a bug in **Apple's MPSGraph framework**, not MetalHLO. When the graph contains integer constant tensors (e.g., `shape [9,2] int32`) that are reshaped (adding size-1 dims: `[9,2] -> [9,2,1,1]`) in the same graph as `dynamic_update_slice` operations (which create `tensor<2xi32>` index concatenations), MPSGraph's internal IR optimization incorrectly fuses the two integer tensors, creating a reshape from `[2]` to `[9,1,1]`.

**Workarounds attempted (all failed):**
- `expandDims` instead of `reshape` — MPSGraph folds back to reshape
- Element count validation in `compileReshape` — bug is internal to MPSGraph, not in our reshape calls
- MLIR call inlining to bypass MPSGraph — O2 path has output memory kind issues

**Recommended path forward:**
1. File Apple Feedback for MPSGraph reshape optimization bug
2. Investigate avoiding integer constant + reshape + DUS pattern in the graph construction
3. Consider generating Metal compute kernels for these operations instead of MPSGraph nodes

## Numerical Accuracy (lid_driven_cavity)

For the working example (2000 timesteps, 100x100 grid):

| Array | Max Abs Diff | Notes |
|-------|-------------|-------|
| f (distribution) | 8.05e-07 | Float32 accumulation drift |
| rho (density) | 1.31e-06 | Excellent |
| u (velocity) | 2.06e-06 | Excellent |

Numerical accuracy within expected float32 tolerances over 2000 iterative steps.

## Changes Made

| File | Changes |
|------|---------|
| `Sources/PJRTMetalHLO/PJRTExecutable.swift` | Pipe deadlock fix, debug compile logging, MLIR call inliner (disabled) |
| `Sources/MetalHLOCore/Compiler/MPSGraphCompiler.swift` | Constant pre-compilation, reshape validation, expandDims workaround (int32) |
| `examples/vivsim/test_poiseuille_channel.py` | Import fix for vivsim submodule compatibility |
