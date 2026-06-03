"""MetalHLO PJRT plugin for JAX.

Registers MetalHLO as a JAX backend, enabling custom Metal kernel execution
on Apple Silicon GPUs via the PJRT plugin interface.
"""

import os
import sys
import jax._src.xla_bridge as xb
import jaxlib.xla_client as xla_client


def initialize():
    """Register the MetalHLO plugin with JAX's PJRT backend system."""
    # The Swift dylib spawns a Python helper subprocess to convert MLIR
    # bytecode → text. ProcessInfo.processInfo.arguments[0] on macOS resolves
    # to the underlying framework binary (e.g. /opt/homebrew/Cellar/.../Python),
    # which doesn't see venv site-packages. Pin to sys.executable so the helper
    # runs in the same interpreter as JAX.
    os.environ.setdefault("METALHLO_PYTHON", sys.executable)

    # JAX 0.10+ rejects re-registering an already-loaded plugin with
    # ALREADY_EXISTS. Skip if a previous explicit register_plugin() call
    # (e.g. from user code) already loaded us.
    if xla_client.pjrt_plugin_loaded("metalhlo"):
        return

    plugin_dir = os.path.dirname(__file__)

    # Look for the dylib in several locations:
    # 1. Alongside this package (installed via pip)
    # 2. In the Swift build output directory
    candidates = [
        os.path.join(plugin_dir, "libPJRTMetalHLO.dylib"),
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", ".build", "release", "libPJRTMetalHLO.dylib",
        ),
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", ".build", "debug", "libPJRTMetalHLO.dylib",
        ),
    ]

    # Allow override via environment variable
    env_path = os.environ.get("METALHLO_PLUGIN_PATH")
    if env_path:
        candidates.insert(0, env_path)

    library_path = None
    for path in candidates:
        resolved = os.path.realpath(path)
        if os.path.isfile(resolved):
            library_path = resolved
            break

    if library_path is None:
        raise RuntimeError(
            "Could not find libPJRTMetalHLO.dylib. "
            "Build the Swift package first with: swift build -c release, "
            "or set METALHLO_PLUGIN_PATH to the dylib location."
        )

    xb.register_plugin(
        "metalhlo",
        priority=500,
        library_path=library_path,
        options=None,
    )

    _register_host_callback_lowerings()
    _register_linalg_lowerings()


def _register_linalg_lowerings():
    """Make JAX emit the LAPACK-FFI StableHLO for linear-algebra primitives on
    the metalhlo platform, reusing JAX's own CPU lowering rules.

    jnp.linalg.svd lowers to `@lapack_sgesdd_ffi` only on the cpu/cuda/rocm
    platforms; for a custom plugin platform like "metalhlo" JAX has no rule and
    falls back to a generic `eigh`-based path (which itself has no metalhlo rule),
    so a jit'd svd raises NotImplementedError at trace time. The MetalHLO Swift
    compiler routes `@lapack_sgesdd_ffi` to a host-side Accelerate LAPACK call, so
    we want exactly the CPU lowering. JAX's `register_lowering` refuses unknown
    platforms, so we copy the CPU lowering entry directly into the per-platform
    lowering table for "metalhlo".

    jnp.linalg.qr decomposes into the `geqrf` (factorization → `@lapack_sgeqrf_ffi`)
    and `householder_product` (materialize Q → `@lapack_sorgqr_ffi`) primitives,
    each with a CPU-only lowering; `qr_p` itself has no platform lowering. Copying
    those two CPU entries makes a jit'd qr emit the LAPACK FFI custom_calls the
    Swift compiler routes to host-side Accelerate LAPACK.
    """
    try:
        from jax._src.interpreters import mlir
        from jax._src.lax import linalg as lax_linalg
    except Exception:
        return

    cpu_table = mlir._platform_specific_lowerings.get("cpu", {})
    metal_table = mlir._platform_specific_lowerings["metalhlo"]
    for prim_name in ("svd_p", "geqrf_p", "householder_product_p"):
        prim = getattr(lax_linalg, prim_name, None)
        if prim is None:
            continue
        entry = cpu_table.get(prim)
        if entry is not None:
            metal_table[prim] = entry


def _register_host_callback_lowerings():
    """Make JAX emit StableHLO for side-effecting host callbacks on metalhlo.

    JAX only registers the lowering rules for jax.debug.print / jax.debug.callback
    on the built-in cpu/gpu/tpu platforms. For a custom PJRT plugin platform like
    "metalhlo", a jit'd function containing jax.debug.print otherwise raises
    NotImplementedError at trace time ("MLIR translation rule for primitive
    'debug_print' not found for platform metalhlo") — before any StableHLO exists.

    On cpu/gpu/tpu these lower to a side-effecting host callback. JAX's own
    lowering rule routes through `emit_python_callback`, which hard-rejects any
    platform outside {cpu, cuda, rocm, tpu}, so it cannot be reused here. Instead
    we register a no-op lowering: both primitives produce no SSA results
    (multiple_results = True, zero outputs), so emitting nothing is a valid
    lowering. The print/callback is silently dropped and the device computation is
    numerically unchanged.

    (The MetalHLO Swift compiler additionally drops the equivalent
    `stablehlo.custom_call @xla_ffi_python_cpu_callback` host-callback op should it
    ever appear in StableHLO fed in directly — e.g. a module lowered for cpu.)

    Result-producing callbacks (jax.pure_callback / io_callback) are intentionally
    NOT touched: they feed real values back into the graph and need a host
    round-trip the plugin does not implement, so they continue to fail loudly.
    """
    try:
        from jax._src.interpreters import mlir
        from jax._src import debugging
    except Exception:
        return

    def _noop_host_callback_lowering(ctx, *args, **params):
        # debug_print / debug_callback have no results; drop the side effect.
        return []

    for prim_name in ("debug_print_p", "debug_callback_p"):
        prim = getattr(debugging, prim_name, None)
        if prim is None:
            continue
        try:
            mlir.register_lowering(prim, _noop_host_callback_lowering, platform="metalhlo")
        except Exception:
            # Lowering registration is best-effort; never block plugin load.
            pass
