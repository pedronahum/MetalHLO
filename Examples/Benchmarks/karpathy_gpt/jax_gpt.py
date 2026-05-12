"""
JAX port of Karpathy's "atomic GPT" — same model, same data, same hand-rolled
Adam, same hyperparameters as `pure_python.py`. The single-position-at-a-time
autograd loop becomes one fixed-shape forward over `block_size` positions with
a lower-triangular causal mask — mathematically identical for valid positions,
and amenable to `jax.jit`.

Usage:
    # JAX CPU backend
    python jax_gpt.py --backend cpu --steps 1000

    # MetalHLO PJRT plugin (Apple GPU)
    python jax_gpt.py --backend metalhlo --steps 1000

Init weights and the document shuffle order are seeded from Python's
`random` to match `pure_python.py` byte-for-byte, so the first training-step
loss should agree between the two runs to fp32 precision; the trajectories
drift after that as floating-point reduction orderings diverge.
"""

import argparse
import os
import random as pyrandom  # same RNG as Karpathy's pure_python.py — Python's `random`, not jax.random
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    choices=["cpu", "metalhlo"],
    default="metalhlo",
    help="Which JAX backend to use.",
)
parser.add_argument("--steps", type=int, default=1000, help="Number of training steps.")
parser.add_argument(
    "--skip-inference",
    action="store_true",
    help="Skip the 20-sample generation loop at the end (training-only timing).",
)
parser.add_argument(
    "--print-every",
    type=int,
    default=50,
    help="How often (in steps) to update the progress line.",
)
args = parser.parse_args()

# Plugin registration must run BEFORE the first `jax.devices()` call.
if args.backend == "metalhlo":
    import jax._src.xla_bridge as xb
    import jaxlib.xla_client as xla_client

    if not xla_client.pjrt_plugin_loaded("metalhlo"):
        repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        candidates = [
            os.path.join(repo_root, ".build", "release", "libPJRTMetalHLO.dylib"),
            os.path.join(repo_root, ".build", "debug", "libPJRTMetalHLO.dylib"),
        ]
        found = next((p for p in candidates if os.path.isfile(p)), None)
        if found is None:
            print(
                "ERROR: libPJRTMetalHLO.dylib not found. "
                "Build the Swift package first: swift build -c release"
            )
            sys.exit(1)
        print(f"Registering MetalHLO PJRT plugin from:\n  {found}")
        xb.register_plugin("metalhlo", priority=500, library_path=found, options=None)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# ─── Data ─────────────────────────────────────────────────────────────

pyrandom.seed(42)  # matches pure_python.py — gives the same shuffle + same init draws

HERE = os.path.dirname(os.path.realpath(__file__))
INPUT_PATH = os.path.join(HERE, "input.txt")
if not os.path.exists(INPUT_PATH):
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, INPUT_PATH)
docs = [line.strip() for line in open(INPUT_PATH) if line.strip()]
pyrandom.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ─── Hyperparameters (mirror pure_python.py exactly) ──────────────────

n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps_adam = 1e-8

# ─── Parameter init — Python random.gauss to match pure_python.py ─────

def init_matrix(nout, nin, std=0.08):
    # Draws happen in the same order as pure_python.py so weights are bit-equal.
    return jnp.array(
        [[pyrandom.gauss(0, std) for _ in range(nin)] for _ in range(nout)],
        dtype=jnp.float32,
    )


# Choose device + materialize params there.
if args.backend == "cpu":
    device = jax.devices("cpu")[0]
else:
    device = jax.devices("metalhlo")[0]
print(f"Backend: {args.backend}")
print(f"Device:  {device}")

with jax.default_device(device):
    params = {
        "wte": init_matrix(vocab_size, n_embd),
        "wpe": init_matrix(block_size, n_embd),
        "lm_head": init_matrix(vocab_size, n_embd),
    }
    for li in range(n_layer):
        params[f"layer{li}.attn_wq"] = init_matrix(n_embd, n_embd)
        params[f"layer{li}.attn_wk"] = init_matrix(n_embd, n_embd)
        params[f"layer{li}.attn_wv"] = init_matrix(n_embd, n_embd)
        params[f"layer{li}.attn_wo"] = init_matrix(n_embd, n_embd)
        params[f"layer{li}.mlp_fc1"] = init_matrix(4 * n_embd, n_embd)
        params[f"layer{li}.mlp_fc2"] = init_matrix(n_embd, 4 * n_embd)
n_params = sum(int(jnp.size(p)) for p in jax.tree.leaves(params))
print(f"num params: {n_params}")

# ─── Model ────────────────────────────────────────────────────────────


def rmsnorm(x):
    # x last dim is n_embd; same eps & formulation as pure_python.py.
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * (ms + 1e-5) ** -0.5


def gpt_forward(params, tokens):
    """Run the whole sequence through GPT in one shot. `tokens` is shape
    (block_size,); positions past the document end attend to junk but the
    loss is masked out, so they don't contribute to the gradient."""
    pos = jnp.arange(block_size)
    x = params["wte"][tokens] + params["wpe"][pos]  # (block_size, n_embd)
    x = rmsnorm(x)

    causal_mask = jnp.tril(jnp.ones((block_size, block_size), dtype=jnp.bool_))

    for li in range(n_layer):
        residual = x
        x = rmsnorm(x)
        # Per-position linear: x @ W^T where W is (n_out, n_in) as in pure_python.
        Q = x @ params[f"layer{li}.attn_wq"].T  # (block_size, n_embd)
        K = x @ params[f"layer{li}.attn_wk"].T
        V = x @ params[f"layer{li}.attn_wv"].T
        # Split into heads: (n_head, block_size, head_dim).
        Q = Q.reshape(block_size, n_head, head_dim).transpose(1, 0, 2)
        K = K.reshape(block_size, n_head, head_dim).transpose(1, 0, 2)
        V = V.reshape(block_size, n_head, head_dim).transpose(1, 0, 2)
        # pure_python divides by sqrt(head_dim) (Karpathy uses `head_dim**0.5`
        # which is the same).
        scores = (Q @ K.transpose(0, 2, 1)) / jnp.sqrt(jnp.float32(head_dim))
        scores = jnp.where(causal_mask, scores, jnp.float32(-1e30))
        attn = jax.nn.softmax(scores, axis=-1)
        out = attn @ V  # (n_head, block_size, head_dim)
        out = out.transpose(1, 0, 2).reshape(block_size, n_embd)
        x = out @ params[f"layer{li}.attn_wo"].T
        x = x + residual

        residual = x
        x = rmsnorm(x)
        x = x @ params[f"layer{li}.mlp_fc1"].T
        x = jax.nn.relu(x)
        x = x @ params[f"layer{li}.mlp_fc2"].T
        x = x + residual

    logits = x @ params["lm_head"].T  # (block_size, vocab_size)
    return logits


def loss_fn(params, inputs, targets, valid_mask):
    """inputs / targets: (block_size,); valid_mask: (block_size,) float — 1 where the
    position is inside the document, 0 for padding."""
    logits = gpt_forward(params, inputs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    per_pos = -log_probs[jnp.arange(block_size), targets]
    return jnp.sum(per_pos * valid_mask) / jnp.sum(valid_mask)


# ─── Adam (hand-rolled to match pure_python.py exactly) ───────────────

opt_state = {
    "m": jax.tree.map(jnp.zeros_like, params),
    "v": jax.tree.map(jnp.zeros_like, params),
}


@jax.jit
def train_step(params, opt_state, inputs, targets, valid_mask, lr_t, step):
    """One Adam update step. `step` is a JAX scalar passed in (not baked as a
    static) so bias-correction (1 - beta**(step+1)) matches pure_python.py
    without re-tracing on every iteration."""
    loss, grads = jax.value_and_grad(loss_fn)(params, inputs, targets, valid_mask)
    step_f = step.astype(jnp.float32)
    bc1 = 1.0 - beta1 ** (step_f + 1.0)
    bc2 = 1.0 - beta2 ** (step_f + 1.0)
    new_m = jax.tree.map(lambda m, g: beta1 * m + (1.0 - beta1) * g, opt_state["m"], grads)
    new_v = jax.tree.map(lambda v, g: beta2 * v + (1.0 - beta2) * g * g, opt_state["v"], grads)
    new_params = jax.tree.map(
        lambda p, m, v: p - lr_t * (m / bc1) / (jnp.sqrt(v / bc2) + eps_adam),
        params,
        new_m,
        new_v,
    )
    return new_params, {"m": new_m, "v": new_v}, loss


# ─── Training loop ────────────────────────────────────────────────────


def tokenize_doc(doc: str) -> tuple[list[int], int]:
    """Encode the doc the same way pure_python.py does and return the
    (block_size+1)-long padded token sequence plus the number of valid
    prediction positions."""
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    padded = tokens[: block_size + 1] + [0] * max(0, block_size + 1 - len(tokens))
    return padded, n


num_steps = args.steps
step_times: list[float] = []
final_loss = float("nan")

print(f"Training for {num_steps} steps on {args.backend}...")
train_start = time.perf_counter()
for step in range(num_steps):
    doc = docs[step % len(docs)]
    padded, n = tokenize_doc(doc)
    inputs = jnp.asarray(padded[:block_size], dtype=jnp.int32)
    targets = jnp.asarray(padded[1:], dtype=jnp.int32)
    # mask is exactly the `(1/n) * sum_{i<n} loss_i` weighting from pure_python.
    valid_mask = (jnp.arange(block_size) < n).astype(jnp.float32)
    lr_t = jnp.float32(learning_rate * (1.0 - step / num_steps))
    step_idx = jnp.int32(step)

    step_start = time.perf_counter()
    params, opt_state, loss = train_step(
        params, opt_state, inputs, targets, valid_mask, lr_t, step_idx
    )
    final_loss = float(loss)  # .item() — forces sync to host
    step_times.append(time.perf_counter() - step_start)

    if (step + 1) % args.print_every == 0 or step + 1 == num_steps:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {final_loss:.4f}", end="\r", flush=True)

train_elapsed = time.perf_counter() - train_start
print()
warmup = max(1, num_steps // 10)
steady = step_times[warmup:] if len(step_times) > warmup else step_times
mean_steady = sum(steady) / max(1, len(steady))
print(
    f"training: {num_steps} steps in {train_elapsed:.2f}s "
    f"(mean step after {warmup}-step warmup: {mean_steady*1000:.2f}ms)"
)
print(f"final training loss: {final_loss:.4f}")

# ─── Inference ────────────────────────────────────────────────────────

if not args.skip_inference:
    temperature = 0.5

    @jax.jit
    def predict_probs(params, tokens, temperature):
        # Compute softmax over all positions in one go and index host-side —
        # MetalHLO's eager-mode `softmax` on a 1D slice currently errors with
        # a memory-kinds/dtypes mismatch, so we stay inside the JIT.
        logits = gpt_forward(params, tokens)
        return jax.nn.softmax(logits / temperature, axis=-1)

    print("--- inference (new, hallucinated names) ---")
    temperature_jax = jnp.float32(temperature)
    for sample_idx in range(20):
        seq = [BOS] + [0] * (block_size - 1)
        sample_chars: list[str] = []
        for pos_id in range(block_size):
            tokens = jnp.asarray(seq[:block_size], dtype=jnp.int32)
            all_probs = predict_probs(params, tokens, temperature_jax)
            probs_at_pos = all_probs[pos_id]
            probs_np = [float(p) for p in probs_at_pos]
            next_id = pyrandom.choices(range(vocab_size), weights=probs_np)[0]
            if next_id == BOS:
                break
            sample_chars.append(uchars[next_id])
            if pos_id + 1 < block_size:
                seq[pos_id + 1] = next_id
        print(f"sample {sample_idx+1:2d}: {''.join(sample_chars)}")
