"""
ResNet18 training on CIFAR-10 using Flax NNX.

Mirrored from tillahoffmann/jax-mps (examples/resnet/main.py), with one
addition: a `--backend {cpu,metalhlo}` flag that registers the
MetalHLO PJRT plugin so the same script can target either CPU or our
backend.

Usage:
    # Train on MetalHLO (GPU)
    python main.py --backend metalhlo

    # Train on CPU for comparison
    python main.py --backend cpu

    # Limit training steps (useful for quick comparisons)
    python main.py --backend metalhlo --steps 30
"""

import argparse
import os
import sys
from time import perf_counter

# Plugin registration must happen BEFORE the first `jax.devices()` call,
# so we do it here, gated by --backend, before importing the rest of jax.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    choices=["cpu", "metalhlo"],
    default="metalhlo",
    help="Which JAX backend to use.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Number of steps (overrides epochs). Useful for quick benchmarks.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
    help="Training batch size.",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-3,
    help="Adam learning rate.",
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

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

# Local imports — keep after backend registration so JAX picks the right device.
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from data import load_cifar10  # noqa: E402
from model import ResNet18  # noqa: E402


def loss_fn(model, inputs, labels_onehot):
    logits = model(inputs)
    return optax.softmax_cross_entropy(logits, labels_onehot).mean()


@nnx.jit
def train_step(model, optimizer, inputs, labels_onehot):
    loss, grads = nnx.value_and_grad(loss_fn)(model, inputs, labels_onehot)
    optimizer.update(model, grads)
    return loss


def main():
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = 1

    if args.backend == "cpu":
        device = jax.devices("cpu")[0]
    else:
        device = jax.devices("metalhlo")[0]
    print(f"Backend: {args.backend}")
    print(f"Device:  {device}")

    print("Loading CIFAR-10...")
    images, labels = load_cifar10()
    num_samples = len(images)
    print(f"Loaded {num_samples:,} training samples")

    with jax.default_device(device):
        model = ResNet18(num_classes=10, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)

    num_batches = num_samples // BATCH_SIZE
    print(f"Preparing {num_batches} batches on device...")
    batched_images = jax.device_put(
        jnp.array(
            images[: num_batches * BATCH_SIZE]
            .reshape(num_batches, BATCH_SIZE, 32, 32, 3)
            .copy()
        ),
        device,
    )
    batched_labels = jax.device_put(
        jax.nn.one_hot(
            labels[: num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE).copy(),
            10,
        ),
        device,
    )

    num_steps = args.steps if args.steps else EPOCHS * num_batches
    batch_idx = 0
    times_per_step = []

    print(f"Starting training for {num_steps} steps...")
    steps = tqdm(range(num_steps))
    loss = jnp.nan
    for _ in steps:
        start = perf_counter()
        loss = train_step(
            model,
            optimizer,
            batched_images[batch_idx],
            batched_labels[batch_idx],
        ).item()
        end = perf_counter()
        times_per_step.append(end - start)
        batch_idx = (batch_idx + 1) % num_batches
        steps.set_description(f"loss = {loss:.3f}")

    print(f"Final training loss: {loss:.3f}")
    # Discard the first half (warm-up: JIT compilation, allocator warmup).
    times_per_step = times_per_step[len(times_per_step) // 2 :]
    mean_step = sum(times_per_step) / len(times_per_step)
    print(f"Time per step (second half, mean over {len(times_per_step)} steps): {mean_step:.3f}s")


if __name__ == "__main__":
    main()
