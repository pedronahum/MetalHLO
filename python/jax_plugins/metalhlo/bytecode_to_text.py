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
    # Strip precision_config and precision attributes (not needed for execution)
    text = re.sub(r",\s*precision_config\s*=\s*\[[^\]]*\]", "", text)
    text = re.sub(r",\s*precision\s*=\s*\[[^\]]*\]", "", text)

    # Convert generic form inherent attributes <{...}> to {...} for parser compatibility
    # Matches ") <{" at end of generic form operand list and "}>" at end of attribute block
    text = text.replace("<{", "{")
    text = re.sub(r"\}>\s*:", "} :", text)

    # Convert dynamic_slice 'sizes' attribute to 'slice_sizes' for parser compatibility
    text = re.sub(r",\s*sizes\s*=\s*\[", ", slice_sizes = [", text)

    # Convert convolution parenthesized operands to standard form
    # stablehlo.convolution(%arg0, %arg1) dim_numbers -> stablehlo.convolution %arg0, %arg1, dim_numbers
    text = re.sub(
        r"stablehlo\.convolution\(([^)]+)\)\s*dim_numbers",
        r"stablehlo.convolution \1, dim_numbers",
        text,
    )
    # Rename convolution dim_numbers to dimension_numbers
    text = re.sub(r"\bdim_numbers\s*=", "dimension_numbers =", text)

    # Extract convolution window= {...} attributes to individual top-level attributes
    def expand_conv_window(m):
        window_body = m.group(1)
        result = ""
        # stride = [...] -> window_strides = [...]
        stride_m = re.search(r"stride\s*=\s*(\[[^\]]*\])", window_body)
        if stride_m:
            result += ", window_strides = " + stride_m.group(1)
        # pad = [...] -> padding = [...]
        pad_m = re.search(r"pad\s*=\s*(\[\[[^\]]*\]\]|\[[^\]]*\])", window_body)
        if pad_m:
            result += ", padding = " + pad_m.group(1)
        # lhs_dilate = [...] -> lhs_dilation = [...]
        lhs_m = re.search(r"lhs_dilate\s*=\s*(\[[^\]]*\])", window_body)
        if lhs_m:
            result += ", lhs_dilation = " + lhs_m.group(1)
        # rhs_dilate = [...] -> rhs_dilation = [...]
        rhs_m = re.search(r"rhs_dilate\s*=\s*(\[[^\]]*\])", window_body)
        if rhs_m:
            result += ", rhs_dilation = " + rhs_m.group(1)
        return result

    text = re.sub(
        r",\s*window\s*=\s*\{([^}]*)\}", expand_conv_window, text
    )

    # Inline convolution trailing attribute block {batch_group_count = N : i64, ...}
    # After window expansion, the block appears right before ": (tensor..."
    # Use [^}\n]* to prevent matching across lines (avoids eating module body)
    def inline_conv_attrs(m):
        attrs_body = m.group(1)
        # Strip type annotations like ": i64"
        attrs_body = re.sub(r"\s*:\s*i\d+", "", attrs_body)
        return ", " + attrs_body.strip()

    text = re.sub(
        r" \{([^}\n]*(?:batch_group_count|feature_group_count)[^}\n]*)\} :",
        lambda m: inline_conv_attrs(m) + " :",
        text,
    )

    # Clean up extra whitespace
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

    sys.stdout.write(text)


if __name__ == "__main__":
    main()
