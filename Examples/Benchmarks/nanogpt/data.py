"""tinyshakespeare loader + char-level tokenizer + random-batch sampler.

Used by the nanoGPT benchmark to feed (batch_size, block_size) windows of
ints into the model. Same dataset Karpathy's nanoGPT uses for the
char-level run; small enough to fit in RAM end-to-end.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

_TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
)


def download(cache_path: str) -> str:
    """Download tinyshakespeare to `cache_path` if not already present."""
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        urllib.request.urlretrieve(_TINYSHAKESPEARE_URL, cache_path)
    with open(cache_path, "r", encoding="utf-8") as f:
        return f.read()


def build_tokenizer(text: str) -> tuple[dict[str, int], list[str]]:
    """Char-level vocabulary built directly from the corpus."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return stoi, chars


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    return np.array([stoi[ch] for ch in text], dtype=np.int32)


def decode(ids: list[int], itos: list[str]) -> str:
    return "".join(itos[i] for i in ids)


class RandomBatchSampler:
    """Samples random (batch_size, block_size+1) windows from a token stream.

    Returns (inputs, targets) where targets are the next-token labels — i.e.
    `targets[b, t] = inputs[b, t + 1]` and the input is the token at that
    position, exactly the standard causal-LM setup.
    """

    def __init__(self, tokens: np.ndarray, *, batch_size: int, block_size: int, seed: int = 0):
        self.tokens = tokens
        self.batch_size = batch_size
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        max_start = len(self.tokens) - self.block_size - 1
        starts = self.rng.integers(0, max_start, size=(self.batch_size,))
        inputs = np.stack([self.tokens[s : s + self.block_size] for s in starts])
        targets = np.stack([self.tokens[s + 1 : s + self.block_size + 1] for s in starts])
        return inputs, targets
