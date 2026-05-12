"""nanoGPT training benchmark — JAX CPU vs MetalHLO.

A 6-layer / 384-dim / 6-head GPT trained char-level on tinyshakespeare,
batch_size=16, block_size=256. ~10.6 M parameters, ~150 GFLOPs per
step — the natural compute-bound counterpart to the per-step-overhead
Karpathy-GPT benchmark in `../karpathy_gpt/`.

Usage:
    # JAX CPU
    python main.py --backend cpu --steps 50

    # MetalHLO PJRT plugin (Apple GPU)
    python main.py --backend metalhlo --steps 50
"""

import argparse
import os
import sys
import time

# Plugin registration must run BEFORE the first `jax.devices()` call.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    choices=["cpu", "metalhlo"],
    default="metalhlo",
    help="Which JAX backend to use.",
)
parser.add_argument("--steps", type=int, default=50, help="Number of training steps.")
parser.add_argument("--batch-size", type=int, default=16, help="Tokens per micro-batch.")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
parser.add_argument(
    "--skip-inference",
    action="store_true",
    help="Skip the post-training sample generation (training-only timing).",
)
parser.add_argument(
    "--gen-tokens",
    type=int,
    default=200,
    help="How many tokens to generate per sample (if inference is enabled).",
)
args = parser.parse_args()

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
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402

# Local imports after plugin registration.
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from data import RandomBatchSampler, build_tokenizer, decode, download, encode  # noqa: E402
from model import GPT  # noqa: E402

# ─── Hyperparameters (nanoGPT char-level config, slightly downsized batch
# so the JAX CPU baseline finishes in a couple of minutes) ─────────────

N_LAYER = 6
N_EMBD = 384
N_HEAD = 6
BLOCK_SIZE = 256


def main():
    # Pick device. Plugin already registered above when needed.
    if args.backend == "cpu":
        device = jax.devices("cpu")[0]
    else:
        device = jax.devices("metalhlo")[0]
    print(f"Backend: {args.backend}")
    print(f"Device:  {device}")

    # ─── Data ────────────────────────────────────────────────────────
    cache_path = os.path.join(HERE, "input.txt")
    text = download(cache_path)
    stoi, itos = build_tokenizer(text)
    vocab_size = len(itos)
    tokens = encode(text, stoi)
    print(f"Loaded tinyshakespeare: {len(text):,} chars, {vocab_size}-token vocab")
    sampler = RandomBatchSampler(
        tokens, batch_size=args.batch_size, block_size=BLOCK_SIZE, seed=0
    )

    # ─── Model + optimizer ───────────────────────────────────────────
    with jax.default_device(device):
        model = GPT(
            vocab_size=vocab_size,
            n_layer=N_LAYER,
            n_embd=N_EMBD,
            n_head=N_HEAD,
            block_size=BLOCK_SIZE,
            rngs=nnx.Rngs(0),
        )
        optimizer = nnx.Optimizer(model, optax.adam(args.learning_rate), wrt=nnx.Param)

    leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
    n_params = sum(int(jnp.size(leaf)) for leaf in leaves if hasattr(leaf, "shape"))
    print(f"num params: {n_params:,}")

    # ─── Train step ──────────────────────────────────────────────────
    def loss_fn(model, inputs, targets):
        logits = model(inputs)  # (B, T, V)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    @nnx.jit
    def train_step(model, optimizer, inputs, targets):
        loss, grads = nnx.value_and_grad(loss_fn)(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    # ─── Training loop ──────────────────────────────────────────────
    times_per_step = []
    final_loss = float("nan")
    print(f"Training for {args.steps} steps "
          f"(batch={args.batch_size}, seq={BLOCK_SIZE}, n_layer={N_LAYER}, n_embd={N_EMBD})...")
    train_start = time.perf_counter()
    for step in range(args.steps):
        inputs_np, targets_np = sampler.sample()
        inputs = jax.device_put(jnp.asarray(inputs_np), device)
        targets = jax.device_put(jnp.asarray(targets_np), device)
        step_start = time.perf_counter()
        loss = train_step(model, optimizer, inputs, targets).item()
        times_per_step.append(time.perf_counter() - step_start)
        final_loss = loss
        if (step + 1) % max(1, args.steps // 10) == 0 or step + 1 == args.steps:
            print(f"step {step+1:4d} / {args.steps:4d} | loss {loss:.4f}", flush=True)

    train_elapsed = time.perf_counter() - train_start
    # Discard first half as warmup (matches the resnet18 benchmark).
    steady = times_per_step[len(times_per_step) // 2 :]
    mean_steady = sum(steady) / max(1, len(steady))
    print(
        f"\ntraining: {args.steps} steps in {train_elapsed:.2f}s "
        f"(mean step over last {len(steady)} steps: {mean_steady*1000:.1f}ms)"
    )
    print(f"final training loss: {final_loss:.4f}")

    # ─── Inference (greedy generation from a single BOS-like context) ─
    if not args.skip_inference:
        print(f"\n--- inference ({args.gen_tokens}-char sample) ---")

        @nnx.jit
        def predict_logits(model, idx):
            return model(idx)

        prompt = "\n"  # newline is the most "BOS-like" token in tinyshakespeare
        prompt_ids = [stoi[ch] for ch in prompt]
        # Maintain a fixed-shape rolling buffer so the JIT signature stays stable.
        ctx = np.zeros((1, BLOCK_SIZE), dtype=np.int32)
        ctx[0, -len(prompt_ids):] = prompt_ids
        out_tokens: list[int] = []
        rng = np.random.default_rng(0)
        for _ in range(args.gen_tokens):
            idx = jax.device_put(jnp.asarray(ctx), device)
            logits = predict_logits(model, idx)  # (1, T, V)
            last_logits = np.asarray(logits[0, -1])  # host-side sampling
            # Temperature 1.0, sample multinomial
            probs = np.exp(last_logits - last_logits.max())
            probs /= probs.sum()
            next_id = int(rng.choice(vocab_size, p=probs))
            out_tokens.append(next_id)
            ctx = np.concatenate([ctx[:, 1:], np.array([[next_id]], dtype=np.int32)], axis=1)

        sample_text = decode(out_tokens, itos)
        # Indent so multi-line samples are visually obvious.
        for line in sample_text.splitlines() or [""]:
            print(f"    {line}")


if __name__ == "__main__":
    main()
