#!/usr/bin/env python3
"""
Per-layer intermediate-value bisection for MetalHLO debugging.

Given a Flax module + variables + inputs, runs the FULL forward pass on
both CPU and MetalHLO (keeping whole-program compilation — the regime
where the remaining bugs manifest) and diffs every Flax-tracked
intermediate. Reports the first layer whose output diverges.

Use this to chase bugs of the "correct in isolation, wrong in full
program" class — #17 (MHA value/kernel grad), #18 (BERT NaN), #19
(CNN drift). The signal we're looking for: an intermediate that
agrees up to layer N and then diverges at layer N+1.

Run examples (from repo root):

    python Examples/FlaxExample/_debug_intermediates.py --case mha-grad
    python Examples/FlaxExample/_debug_intermediates.py --case bert-loss
"""

import argparse
import os
import sys

import numpy as np

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import jax._src.xla_bridge as xb
import jaxlib.xla_client as xla_client

# ── Plugin registration ──────────────────────────────────────────────

if not xla_client.pjrt_plugin_loaded("metalhlo"):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    found = next((p for p in [
        os.path.join(repo_root, ".build", "release", "libPJRTMetalHLO.dylib"),
        os.path.join(repo_root, ".build", "debug", "libPJRTMetalHLO.dylib"),
    ] if os.path.isfile(p)), None)
    if found is None:
        print("ERROR: libPJRTMetalHLO.dylib not found.")
        sys.exit(1)
    xb.register_plugin("metalhlo", priority=500, library_path=found, options=None)

cpu = jax.devices("cpu")[0]
mhlo = jax.devices("metalhlo")[0]


def to_device(tree, dev):
    return jax.tree.map(lambda a: jax.device_put(a, dev), tree)


def flatten_intermediates(d, prefix=""):
    """Flatten nested intermediate dicts to {flat_path: array} entries."""
    out = {}
    if isinstance(d, dict) or isinstance(d, flax.core.FrozenDict):
        for k, v in d.items():
            sub = flatten_intermediates(v, f"{prefix}/{k}")
            out.update(sub)
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            sub = flatten_intermediates(v, f"{prefix}[{i}]")
            out.update(sub)
    else:
        # Leaf (a jax array)
        out[prefix] = d
    return out


def diff_intermediates(cpu_inter, mhlo_inter, *, tol=1e-3):
    """Compare two intermediate dicts. Returns ordered list of
    (path, cpu_array, mhlo_array, max_abs_diff, status)."""
    cpu_flat = flatten_intermediates(cpu_inter)
    mhlo_flat = flatten_intermediates(mhlo_inter)
    keys = sorted(set(cpu_flat) | set(mhlo_flat))
    rows = []
    for k in keys:
        if k not in cpu_flat:
            rows.append((k, None, mhlo_flat[k], float("inf"), "MHLO-ONLY"))
            continue
        if k not in mhlo_flat:
            rows.append((k, cpu_flat[k], None, float("inf"), "CPU-ONLY"))
            continue
        c = np.asarray(cpu_flat[k]).astype(np.float64)
        m = np.asarray(mhlo_flat[k]).astype(np.float64)
        if c.shape != m.shape:
            rows.append((k, c, m, float("inf"), f"SHAPE {c.shape} vs {m.shape}"))
            continue
        if np.any(np.isnan(m)) and not np.any(np.isnan(c)):
            rows.append((k, c, m, float("nan"), "NAN"))
            continue
        d = float(np.abs(c - m).max())
        status = "ok" if d <= tol else "DIFF"
        rows.append((k, c, m, d, status))
    return rows


def print_report(rows, tol):
    print(f"  {'status':>10} {'max abs diff':>14}   path")
    print("  " + "-" * 78)
    first_bad = None
    for path, c, m, d, status in rows:
        flag = "  " if status == "ok" else "* "
        d_str = f"{d:.6e}" if not np.isnan(d) and not np.isinf(d) else f"{d}"
        print(f"  {flag}{status:>8} {d_str:>14}   {path}")
        if status != "ok" and first_bad is None:
            first_bad = (path, status, d)
    print()
    if first_bad:
        path, status, d = first_bad
        print(f"FIRST DIVERGENT: {path}  (status={status}, max abs diff={d})")
    else:
        print(f"All intermediates agree within tol={tol}")


# ── Cases ────────────────────────────────────────────────────────────


def case_mha_forward():
    print("Case: MHA forward — locate first divergent layer")
    print("-" * 60)

    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.MultiHeadDotProductAttention(
                num_heads=4, qkv_features=16, deterministic=True
            )(x)

    np.random.seed(4)
    x = np.random.randn(2, 6, 16).astype(np.float32)
    with jax.default_device(cpu):
        params = M().init(jax.random.PRNGKey(4), jnp.asarray(x))
        out_c, state_c = M().apply(
            params, jnp.asarray(x), capture_intermediates=True
        )
    params_m = to_device(params, mhlo)
    out_m, state_m = M().apply(
        params_m, jax.device_put(jnp.asarray(x), mhlo), capture_intermediates=True
    )
    # capture_intermediates returns ({'intermediates': {...}})
    rows = diff_intermediates(state_c, state_m, tol=1e-3)
    print_report(rows, tol=1e-3)


def case_mha_grad():
    print("Case: MHA grad — locate first divergent grad component")
    print("-" * 60)

    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.MultiHeadDotProductAttention(
                num_heads=4, qkv_features=16, deterministic=True
            )(x)

    np.random.seed(4)
    x = np.random.randn(2, 6, 16).astype(np.float32)
    y = np.random.randn(2, 6, 16).astype(np.float32)

    def loss(p, x_, y_):
        return jnp.mean((M().apply(p, x_) - y_) ** 2)

    with jax.default_device(cpu):
        params = M().init(jax.random.PRNGKey(4), jnp.asarray(x))
        out_c, state_c = M().apply(
            params, jnp.asarray(x), capture_intermediates=True
        )
        g_c = jax.grad(loss)(params, jnp.asarray(x), jnp.asarray(y))

    params_m = to_device(params, mhlo)
    out_m, state_m = M().apply(
        params_m, jax.device_put(jnp.asarray(x), mhlo), capture_intermediates=True
    )
    g_m = jax.grad(loss)(
        params_m, jax.device_put(jnp.asarray(x), mhlo),
        jax.device_put(jnp.asarray(y), mhlo),
    )

    print("Forward intermediates:")
    rows = diff_intermediates(state_c, state_m, tol=1e-4)
    print_report(rows, tol=1e-4)

    print()
    print("Grad components:")
    grad_rows = diff_intermediates(g_c, g_m, tol=1e-3)
    print_report(grad_rows, tol=1e-3)

    # Backward-chain bisection: identify whether the upstream of the
    # value-kernel grad (dV = grad w.r.t. value-projection output) is
    # already wrong, or only dW_v = dV @ x is wrong.
    print()
    print("Backward-chain bisection — dV (grad w.r.t. value-proj output):")

    # The standard attention pattern:
    #   V = einsum('bsf,fho->bsho', x, W_v)   (per-head value projection)
    # Compute dV by manually unrolling through softmax(QK)V.
    # Easier: use jax.vjp to get the cotangent flowing into V.
    def value_proj(W_v, x):
        # x: (B, S, F_in), W_v: (F_in, H, D) — same shapes as Flax stores
        return jnp.einsum("bsf,fhd->bshd", x, W_v)

    def attention_from_V(V, params_partial, x_, y_):
        """Reconstruct attention using a custom V tensor — lets us pluck
        dV out via jax.grad."""
        # Pull out Q, K projections
        q = jnp.einsum(
            "bsf,fhd->bshd", x_, params_partial["query"]["kernel"]
        ) + params_partial["query"]["bias"]
        k = jnp.einsum(
            "bsf,fhd->bshd", x_, params_partial["key"]["kernel"]
        ) + params_partial["key"]["bias"]
        v = V + params_partial["value"]["bias"]
        # Attention scores
        scores = jnp.einsum("bshd,bthd->bhst", q, k) / jnp.sqrt(q.shape[-1])
        weights = jax.nn.softmax(scores, axis=-1)
        attn = jnp.einsum("bhst,bthd->bshd", weights, v)
        # Output projection
        out = jnp.einsum(
            "bshd,hdo->bso", attn, params_partial["out"]["kernel"]
        ) + params_partial["out"]["bias"]
        return jnp.mean((out - y_) ** 2)

    with jax.default_device(cpu):
        # Compute V (pre-bias) on CPU
        V_c = value_proj(
            params["params"]["MultiHeadDotProductAttention_0"]["value"]["kernel"],
            jnp.asarray(x),
        )
        dV_c = jax.grad(attention_from_V)(
            V_c,
            params["params"]["MultiHeadDotProductAttention_0"],
            jnp.asarray(x),
            jnp.asarray(y),
        )

    V_m = value_proj(
        params_m["params"]["MultiHeadDotProductAttention_0"]["value"]["kernel"],
        jax.device_put(jnp.asarray(x), mhlo),
    )
    dV_m = jax.grad(attention_from_V)(
        V_m,
        params_m["params"]["MultiHeadDotProductAttention_0"],
        jax.device_put(jnp.asarray(x), mhlo),
        jax.device_put(jnp.asarray(y), mhlo),
    )
    dV_diff = float(np.abs(np.asarray(dV_c) - np.asarray(dV_m)).max())
    V_diff = float(np.abs(np.asarray(V_c) - np.asarray(V_m)).max())
    print(f"  V forward (no bias) diff: {V_diff:.6e}")
    print(f"  dV (cotangent into V)  diff: {dV_diff:.6e}")
    if dV_diff > 1e-3:
        print(
            "  dV already wrong → bug is upstream of the V kernel grad "
            "(somewhere in the softmax/attn backward chain)"
        )
    else:
        print(
            "  dV correct → bug is in dW_v = x^T · dV (the V-projection "
            "kernel-grad dot_general specifically)"
        )


def case_bert_loss():
    print("Case: BERT forward + loss — show cross-call NaN flakiness")
    print("-" * 60)

    class MiniBERT(nn.Module):
        vocab: int = 32
        d_model: int = 32
        num_heads: int = 4
        d_ff: int = 64
        num_classes: int = 3
        seq_len: int = 8

        @nn.compact
        def __call__(self, tokens):
            tok_emb = nn.Embed(self.vocab, self.d_model)(tokens)
            pos_emb = self.param(
                "pos_emb", nn.initializers.normal(0.02), (self.seq_len, self.d_model)
            )
            x = tok_emb + pos_emb[None]
            h = nn.LayerNorm()(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, qkv_features=self.d_model,
                deterministic=True,
            )(h)
            x = x + h
            h = nn.LayerNorm()(x)
            h = nn.Dense(self.d_ff)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.d_model)(h)
            x = x + h
            return nn.Dense(self.num_classes)(jnp.mean(x, axis=1))

    np.random.seed(0)
    tokens = np.random.randint(0, 32, (4, 8)).astype(np.int32)
    y = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, 4)]

    model = MiniBERT()
    with jax.default_device(cpu):
        params = model.init(jax.random.PRNGKey(0), jnp.asarray(tokens))

    # 1) Pure forward, capture intermediates, run multiple times
    params_m = to_device(params, mhlo)
    tm = jax.device_put(jnp.asarray(tokens), mhlo)
    ym = jax.device_put(jnp.asarray(y), mhlo)
    print("(a) Bare forward, capture_intermediates=True, 3 calls in a row:")
    for i in range(3):
        out_m, state_m = model.apply(
            params_m, tm, capture_intermediates=True
        )
        nan = bool(np.any(np.isnan(np.asarray(out_m))))
        print(f"  call {i}: nan={nan}, output sum = {float(np.asarray(out_m).sum()):.6f}")

    # 2) Bare forward (no capture) — same flaky?
    print("(b) Bare forward, no capture, 3 calls in a row:")
    for i in range(3):
        out_m = model.apply(params_m, tm)
        nan = bool(np.any(np.isnan(np.asarray(out_m))))
        print(f"  call {i}: nan={nan}, output sum = {float(np.asarray(out_m).sum()):.6f}")

    # 3) loss wrapper, 3 calls
    print("(c) loss_fn (forward + log_softmax + sum + mean), 3 calls:")
    def loss_fn(p, t, y_):
        return -jnp.mean(jnp.sum(y_ * jax.nn.log_softmax(model.apply(p, t), -1), -1))
    for i in range(3):
        val = float(loss_fn(params_m, tm, ym))
        print(f"  call {i}: loss = {val}")

    # 4) jit'd loss wrapper, 3 calls
    print("(d) jax.jit(loss_fn), 3 calls:")
    loss_jit = jax.jit(loss_fn)
    for i in range(3):
        val = float(loss_jit(params_m, tm, ym))
        print(f"  call {i}: loss = {val}")


def case_cnn_grad():
    print("Case: Mini-CNN grad — locate first divergent grad component")
    print("-" * 60)

    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Conv(8, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            return nn.Dense(4)(x.reshape((x.shape[0], -1)))

    np.random.seed(2)
    x = np.random.randn(4, 8, 8, 3).astype(np.float32)
    y = np.eye(4, dtype=np.float32)[np.random.randint(0, 4, 4)]

    def loss_fn(p, x_, y_):
        return -jnp.mean(jnp.sum(y_ * jax.nn.log_softmax(M().apply(p, x_), -1), -1))

    with jax.default_device(cpu):
        params = M().init(jax.random.PRNGKey(2), jnp.asarray(x))
        g_c = jax.jit(jax.grad(loss_fn))(
            params, jnp.asarray(x), jnp.asarray(y)
        )

    params_m = to_device(params, mhlo)
    g_m = jax.jit(jax.grad(loss_fn))(
        params_m,
        jax.device_put(jnp.asarray(x), mhlo),
        jax.device_put(jnp.asarray(y), mhlo),
    )
    rows = diff_intermediates(g_c, g_m, tol=1e-3)
    print_report(rows, tol=1e-3)


CASES = {
    "mha-forward": case_mha_forward,
    "mha-grad": case_mha_grad,
    "bert-loss": case_bert_loss,
    "cnn-grad": case_cnn_grad,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=list(CASES.keys()) + ["all"],
        default="all",
        help="Which debug case to run",
    )
    args = parser.parse_args()
    if args.case == "all":
        for name, fn in CASES.items():
            print(f"\n{'=' * 70}")
            fn()
    else:
        CASES[args.case]()
