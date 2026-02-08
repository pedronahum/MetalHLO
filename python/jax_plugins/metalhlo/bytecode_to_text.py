#!/usr/bin/env python3
"""Convert MLIR bytecode (stdin) to MLIR text (stdout).

Used by the MetalHLO PJRT plugin to deserialize MLIR bytecode sent by JAX.
JAX serializes StableHLO programs as VHLO portable artifacts. This script
deserializes them back to standard StableHLO text and strips JAX-specific
annotations that the MetalHLO parser doesn't handle.
"""
import re
import sys


def main():
    bytecode = sys.stdin.buffer.read()
    from jax._src.interpreters.mlir import make_ir_context
    from jaxlib.mlir._mlir_libs import _stablehlo

    ctx = make_ir_context()
    module = _stablehlo.deserialize_portable_artifact(ctx, bytecode)
    text = str(module)

    # Strip module attributes
    text = re.sub(r"(module @\S+)\s+attributes\s+\{[^}]*\}", r"\1", text)
    # Strip 'public' visibility
    text = text.replace("func.func public ", "func.func ")
    # Strip result annotations
    text = re.sub(r'\{jax\.\w+\s*=\s*"[^"]*"\}', "", text)
    # Strip SDY sharding mesh declarations and annotations
    text = re.sub(
        r"^\s*sdy\.mesh\s+@\S+\s*=\s*<[^>]*>\s*$", "", text, flags=re.MULTILINE
    )
    text = re.sub(r"\{sdy\.sharding\s*=\s*#sdy\.sharding<[^>]*>\}", "", text)
    # Clean up extra whitespace
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

    sys.stdout.write(text)


if __name__ == "__main__":
    main()
