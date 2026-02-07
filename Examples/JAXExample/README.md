# JAX + MetalHLO Example

Run JAX computations on Apple Silicon GPUs via the MetalHLO PJRT plugin.

## Setup

```bash
# 1. Build the PJRT plugin
cd <repo-root>
swift build -c release --product PJRTMetalHLO

# 2. Install JAX and the MetalHLO plugin package
pip install jax
pip install -e python/

# 3. Run the example
python Examples/JAXExample/jax_metalhlo_example.py
```

## What It Demonstrates

| Example | Operations | StableHLO Ops Used |
|---------|------------|--------------------|
| Vector arithmetic | `x + y` | `stablehlo.add` |
| Matrix multiply | `A @ B` | `stablehlo.dot_general` |
| JIT-compiled MLP layer | `relu(x @ w + b)` | `dot_general`, `add`, `broadcast`, `maximum` |
| Element-wise math | `exp`, `sqrt`, `tanh` | `stablehlo.exponential`, `sqrt`, `tanh` |
| Reductions | `sum`, `max`, `mean` | `stablehlo.reduce` |
| Softmax | `exp(x - max) / sum(...)` | Pattern fused by MetalHLO's O2 optimizer |
