# vivsim Examples on MetalHLO

These examples are adapted from the [vivsim](https://github.com/haimingz/vivsim) library
to test MetalHLO's JAX PJRT backend support. They cover Lattice Boltzmann Method (LBM)
simulations including cavity flow, channel flow, immersed boundary, and fluid-structure
interaction — exercising a wide range of HLO operations via JAX.

## Prerequisites

1. Build the MetalHLO PJRT plugin:
   ```bash
   cd /path/to/MetalHLO
   swift build -c release
   ```

2. Install vivsim:
   ```bash
   pip install vivsim
   ```

3. Install the MetalHLO JAX plugin (editable mode):
   ```bash
   cd /path/to/MetalHLO/python
   pip install -e .
   ```

4. Verify the plugin loads:
   ```bash
   python -c "import jax; import jax_plugins.metalhlo; print('OK')"
   ```

## Running the Examples (Interactive)

Each example can be run directly with live matplotlib visualization:
```bash
python examples/vivsim/lid_driven_cavity.py
python examples/vivsim/poiseuille_channel.py
python examples/vivsim/flow_pass_cylinder.py
python examples/vivsim/flow_through_text.py
python examples/vivsim/vortex_induced_vibration.py
python examples/vivsim/vortex_induced_vibration_multigrid.py
python examples/vivsim/vortex_induced_vibration_refinement.py
```

Note: `vortex_induced_vibration_multigpu.py` requires multi-device sharding which
MetalHLO does not currently support. It is included for reference only.

## Running the Comparison Benchmark (CPU vs MetalHLO)

The `test_*.py` scripts are reduced-size versions of each example designed for
automated comparison. They run fewer timesteps on a smaller grid and dump all
numerical results as `.npy` files for bitwise comparison.

### Run all benchmarks
```bash
python examples/vivsim/run_comparison.py
```

### Run a single benchmark
```bash
python examples/vivsim/run_comparison.py test_lid_driven_cavity
```

### Available benchmarks

| Script | What it tests |
|--------|---------------|
| `test_lid_driven_cavity` | BGK collision, NEE boundary conditions, streaming |
| `test_poiseuille_channel` | 4 collision/forcing combos (BGK+EDM, BGK+Guo, MRT+Guo, KBC+EDM), body forces |
| `test_flow_pass_cylinder` | Immersed boundary method, `dynamic_slice`/`dynamic_update_slice`, stencil ops |
| `test_flow_through_text` | Boolean masks, bounce-back obstacles, KBC collision |
| `test_vortex_induced_vibration` | Fluid-structure interaction, Newmark integration, dynamic IB region |

### How it works

For each benchmark, `run_comparison.py`:
1. Runs the test script with `JAX_PLATFORMS=cpu` and saves `.npy` arrays to `results/<name>/cpu/`
2. Runs the same script with `JAX_PLATFORMS=metalhlo,cpu` and saves to `results/<name>/metalhlo/`
3. Compares every `.npy` file **bitwise** (exact `np.array_equal`), also reporting max absolute/relative differences
4. Writes a per-test `comparison.json` and an overall `results/summary.json`

### Reading results

```
results/
  summary.json                          # Overall pass/fail + timing for all tests
  test_lid_driven_cavity/
    cpu/                                # All .npy outputs from CPU run
      f.npy, rho.npy, u.npy, ...
      stdout.txt, stderr.txt
    metalhlo/                           # All .npy outputs from MetalHLO run
      f.npy, rho.npy, u.npy, ...
      stdout.txt, stderr.txt
    comparison.json                     # Per-array exact_match + max_abs_diff + max_rel_diff
  test_poiseuille_channel/
    ...
```

### Manual single-backend run

You can run any test script manually by setting environment variables:

```bash
# Run on CPU only
METALHLO_TEST_BACKEND=cpu \
METALHLO_TEST_OUTPUT_DIR=./my_results/cpu \
python examples/vivsim/test_lid_driven_cavity.py

# Run on MetalHLO
METALHLO_TEST_BACKEND=metalhlo \
METALHLO_TEST_OUTPUT_DIR=./my_results/metalhlo \
python examples/vivsim/test_lid_driven_cavity.py
```

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `JAX_PLATFORMS` | `cpu`, `metalhlo,cpu`, `metal,cpu` | Selects the JAX backend. The interactive examples set this automatically. |
| `METALHLO_TEST_BACKEND` | `cpu`, `metalhlo` | Used by `test_*.py` scripts to select backend. Sets `JAX_PLATFORMS` internally. |
| `METALHLO_TEST_OUTPUT_DIR` | path | Where `test_*.py` scripts save `.npy` output files. |
| `METALHLO_TEST_NO_PLOT` | `1` | Disables matplotlib plots (used by `run_comparison.py`). |
| `METALHLO_PLUGIN_PATH` | path to `.dylib` | Override the MetalHLO PJRT plugin library path. |

## Modifications from Upstream vivsim

- Added MetalHLO PJRT backend selection via `JAX_PLATFORMS=metalhlo,metal,cpu` at the top of each file (falls back gracefully if MetalHLO is not available)
- Added backend/device diagnostic prints at startup
- Added "(MetalHLO backend)" to plot titles for identification
- Multi-GPU example includes a note that it runs single-device on MetalHLO
