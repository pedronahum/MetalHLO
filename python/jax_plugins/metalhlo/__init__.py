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
