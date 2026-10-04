"""NumPy inference for Junshi Wang's exported seven-PMT position models."""

from pathlib import Path

import numpy as np


class JunshiReconstructor:
    """Predict x/y in model coordinates, in mm, from seven PMT counts."""

    radius_mm = 39.5

    def __init__(self, variant="uniform"):
        model_path = Path(__file__).with_name("models") / f"junshi_{variant}.npz"
        with np.load(model_path, allow_pickle=False) as model:
            self.mean = model["mean"]
            self.scale = model["scale"]
            self.coefs = tuple(model[f"coefs_{i}"] for i in range(4))
            self.intercepts = tuple(model[f"intercepts_{i}"] for i in range(4))

    def predict(self, counts):
        """Return (N, 2) positions without rotation or radius clipping."""
        counts = np.atleast_2d(np.asarray(counts, dtype=float))
        result = np.empty((len(counts), 2), dtype=float)
        for start in range(0, len(counts), 256):
            x = counts[start : start + 256]
            x = x / np.maximum(x.sum(axis=1, keepdims=True), 1)
            x = (x - self.mean) / self.scale
            for layer, (coefs, intercepts) in enumerate(
                zip(self.coefs, self.intercepts, strict=True)
            ):
                x = x @ coefs + intercepts
                if layer < 3:
                    np.maximum(x, 0, out=x)
            result[start : start + 256] = x
        return result
