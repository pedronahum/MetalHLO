#!/usr/bin/env python3
"""
Run vivsim examples on both CPU and MetalHLO backends, save results, and compare.

Usage:
    python examples/vivsim/run_comparison.py [example_name]

If no example_name is given, runs all examples sequentially.
"""

import subprocess
import sys
import os
import json
import time

PYTHON = "/Users/pedro/miniforge3/bin/python"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EXAMPLES = [
    "test_lid_driven_cavity",
    "test_poiseuille_channel",
    "test_flow_pass_cylinder",
    "test_flow_through_text",
    "test_vortex_induced_vibration",
]


def run_example(name, backend):
    """Run a test script with a specific backend, return (success, elapsed, stderr)."""
    script = os.path.join(SCRIPT_DIR, f"{name}.py")
    out_dir = os.path.join(RESULTS_DIR, name, backend)
    os.makedirs(out_dir, exist_ok=True)

    env = os.environ.copy()
    env["METALHLO_TEST_BACKEND"] = backend
    env["METALHLO_TEST_OUTPUT_DIR"] = out_dir
    # Disable plotting
    env["METALHLO_TEST_NO_PLOT"] = "1"

    print(f"\n{'='*60}")
    print(f"  Running: {name} | Backend: {backend}")
    print(f"  Output:  {out_dir}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [PYTHON, script],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max per example
    )
    elapsed = time.time() - start

    # Save stdout/stderr
    with open(os.path.join(out_dir, "stdout.txt"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(out_dir, "stderr.txt"), "w") as f:
        f.write(result.stderr)

    success = result.returncode == 0
    if not success:
        print(f"  FAILED (exit code {result.returncode})")
        print(f"  stderr (last 500 chars): ...{result.stderr[-500:]}")
    else:
        print(f"  OK ({elapsed:.1f}s)")

    return success, elapsed


def compare_results(name):
    """Compare CPU vs MetalHLO .npy files for exact match."""
    cpu_dir = os.path.join(RESULTS_DIR, name, "cpu")
    mhlo_dir = os.path.join(RESULTS_DIR, name, "metalhlo")

    import numpy as np

    cpu_files = sorted(f for f in os.listdir(cpu_dir) if f.endswith(".npy"))
    mhlo_files = sorted(f for f in os.listdir(mhlo_dir) if f.endswith(".npy"))

    if not cpu_files:
        print(f"  WARNING: No .npy files in CPU output for {name}")
        return False

    if set(cpu_files) != set(mhlo_files):
        print(f"  MISMATCH: Different output files")
        print(f"    CPU:      {cpu_files}")
        print(f"    MetalHLO: {mhlo_files}")
        return False

    all_match = True
    summary = {}
    for fname in cpu_files:
        cpu_arr = np.load(os.path.join(cpu_dir, fname))
        mhlo_arr = np.load(os.path.join(mhlo_dir, fname))

        if cpu_arr.shape != mhlo_arr.shape:
            print(f"  {fname}: SHAPE MISMATCH cpu={cpu_arr.shape} mhlo={mhlo_arr.shape}")
            all_match = False
            summary[fname] = {"status": "SHAPE_MISMATCH"}
            continue

        # Exact bitwise comparison
        exact = np.array_equal(cpu_arr, mhlo_arr)

        # Also compute numerical distance for reporting
        if np.issubdtype(cpu_arr.dtype, np.floating):
            max_abs_diff = float(np.max(np.abs(cpu_arr - mhlo_arr)))
            rel_diff = float(np.max(np.abs(cpu_arr - mhlo_arr) / (np.abs(cpu_arr) + 1e-30)))
            any_nan_cpu = bool(np.any(np.isnan(cpu_arr)))
            any_nan_mhlo = bool(np.any(np.isnan(mhlo_arr)))
        else:
            max_abs_diff = float(np.max(np.abs(cpu_arr.astype(float) - mhlo_arr.astype(float))))
            rel_diff = 0.0
            any_nan_cpu = False
            any_nan_mhlo = False

        status = "EXACT_MATCH" if exact else "DIFFERS"
        if not exact:
            all_match = False

        summary[fname] = {
            "status": status,
            "shape": list(cpu_arr.shape),
            "dtype": str(cpu_arr.dtype),
            "exact_match": exact,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": rel_diff,
            "cpu_nan": any_nan_cpu,
            "mhlo_nan": any_nan_mhlo,
        }

        marker = "OK" if exact else "DIFF"
        print(f"  {fname}: {marker}  "
              f"(max_abs={max_abs_diff:.2e}, max_rel={rel_diff:.2e}, "
              f"shape={cpu_arr.shape}, dtype={cpu_arr.dtype})")

    # Save comparison summary
    comp_path = os.path.join(RESULTS_DIR, name, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Comparison saved to: {comp_path}")

    return all_match


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else EXAMPLES

    results = {}

    for name in selected:
        if name not in EXAMPLES:
            print(f"Unknown example: {name}")
            continue

        # Run CPU first, then MetalHLO
        cpu_ok, cpu_time = run_example(name, "cpu")
        if not cpu_ok:
            results[name] = {"status": "CPU_FAILED"}
            continue

        mhlo_ok, mhlo_time = run_example(name, "metalhlo")
        if not mhlo_ok:
            results[name] = {"status": "METALHLO_FAILED", "cpu_time": cpu_time}
            continue

        print(f"\n--- Comparing: {name} ---")
        match = compare_results(name)
        results[name] = {
            "status": "EXACT_MATCH" if match else "DIFFERS",
            "cpu_time": cpu_time,
            "mhlo_time": mhlo_time,
            "speedup": cpu_time / mhlo_time if mhlo_time > 0 else 0,
        }

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        status = r["status"]
        extra = ""
        if "cpu_time" in r and "mhlo_time" in r:
            extra = f"  cpu={r['cpu_time']:.1f}s  mhlo={r['mhlo_time']:.1f}s  speedup={r.get('speedup',0):.2f}x"
        print(f"  {name}: {status}{extra}")

    # Save overall summary
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
