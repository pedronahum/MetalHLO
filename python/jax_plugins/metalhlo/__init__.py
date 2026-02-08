"""MetalHLO PJRT plugin for JAX.

Registers MetalHLO as a JAX backend, enabling custom Metal kernel execution
on Apple Silicon GPUs via the PJRT plugin interface.
"""

import os
import jax._src.xla_bridge as xb


def initialize():
    """Register the MetalHLO plugin with JAX's PJRT backend system."""
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
