"""Jun: the JW_XiHuTPC_MC v3 LRF maximum-likelihood XY reconstruction.

The bundled response and QE describe the source simulation. Input columns follow
Geant4 copyNo 0..6; detector channels must be mapped to that order by the caller.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_MODEL_DIR = Path(__file__).with_name("models")
DEFAULT_QE = tuple(json.loads((_MODEL_DIR / "jun.json").read_text(encoding="utf-8"))["qe"])


def _load_lrf(path):
    """Load x,y,f0..f6 and fill disk-exterior cells from the nearest valid cell."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    x, y = np.unique(data["x"]), np.unique(data["y"])
    values = np.column_stack([data[f"f{i}"] for i in range(7)])
    values = np.where(np.isnan(values), 0.0, values)
    f = np.zeros((7, len(x), len(y)))
    f[:, np.searchsorted(x, data["x"]), np.searchsorted(y, data["y"])] = values.T
    zero = f.sum(axis=0) == 0
    if zero.any():
        coords = np.array([(xi, yi) for xi in x for yi in y])
        nonzero_indices = np.flatnonzero(~zero.ravel())
        zero_indices = np.flatnonzero(zero.ravel())
        tree = cKDTree(coords[nonzero_indices])
        _, nearest = tree.query(coords[zero_indices])
        flat = f.reshape(7, -1)
        flat[:, zero_indices] = flat[:, nonzero_indices[nearest]]
    return x, y, f


def _interpolate_lrf(grid, xs, ys):
    """Use the source's bilinear interpolation and edge clipping."""
    xg, yg, f = grid
    i = np.clip(np.searchsorted(xg, xs, side="right") - 1, 0, len(xg) - 2)
    j = np.clip(np.searchsorted(yg, ys, side="right") - 1, 0, len(yg) - 2)
    x0, x1, y0, y1 = xg[i], xg[i + 1], yg[j], yg[j + 1]
    tx = np.clip((xs - x0) / np.maximum(x1 - x0, 1e-12), 0, 1)
    ty = np.clip((ys - y0) / np.maximum(y1 - y0, 1e-12), 0, 1)
    f00, f10, f01, f11 = f[:, i, j], f[:, i + 1, j], f[:, i, j + 1], f[:, i + 1, j + 1]
    return (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty) + f01 * (1 - tx) * ty + f11 * tx * ty).T


class JunReconstructor:
    """Reconstruct nonnegative, positive-total seven-PMT PE patterns in mm.

    ``lrf_path`` accepts the source CSV format. ``qe`` supplies seven relative
    quantum efficiencies; omission selects the documented simulation values.
    The model returns positions only, with no fitted uncertainty or quality score.
    """

    radius_mm = 40.0

    def __init__(self, lrf_path=None, qe=None):
        grid = _load_lrf(_MODEL_DIR / "jun_lrf_v3.csv" if lrf_path is None else lrf_path)
        qe = np.asarray(DEFAULT_QE if qe is None else qe, dtype=float)
        self._gx = np.arange(-40.0, 40.5, 1.0)
        self._gy = self._gx.copy()
        x, y = np.meshgrid(self._gx, self._gy)
        response = _interpolate_lrf(grid, x.ravel(), y.ravel()).reshape(81, 81, 7)
        response *= qe[None, None, :]
        total = response.sum(axis=2)
        self._bad = (total <= 1e-12) | (np.hypot(x, y) > self.radius_mm)
        probabilities = np.divide(
            response,
            total[..., None],
            out=np.zeros_like(response),
            where=total[..., None] > 1e-12,
        )
        self._lnp = np.log(np.maximum(probabilities, 1e-300))

    def predict(self, counts):
        """Return (n,2) XY/mm for (n,7) PE counts, scanning 256 events per chunk."""
        counts = np.asarray(counts, dtype=float)
        result = np.empty((len(counts), 2))
        gx, gy = self._gx, self._gy
        for start in range(0, len(counts), 256):
            sl = slice(start, min(start + 256, len(counts)))
            cost = -np.einsum("ek,ijk->eij", counts[sl], self._lnp)
            cost[:, self._bad] = np.inf
            n = len(cost)
            j, i = np.unravel_index(np.argmin(cost.reshape(n, -1), axis=1), self._lnp.shape[:2])
            e = np.arange(n)
            x, y = gx[i].copy(), gy[j].copy()
            im, ip = np.clip(i - 1, 0, len(gx) - 1), np.clip(i + 1, 0, len(gx) - 1)
            denx = cost[e, j, im] - 2 * cost[e, j, i] + cost[e, j, ip]
            m = (
                (i > 0)
                & (i < len(gx) - 1)
                & (denx > 0)
                & np.isfinite(cost[e, j, im])
                & np.isfinite(cost[e, j, ip])
            )
            x[m] += (
                0.5
                * (gx[i[m] + 1] - gx[i[m]])
                * (cost[e[m], j[m], i[m] - 1] - cost[e[m], j[m], i[m] + 1])
                / denx[m]
            )
            jm, jp = np.clip(j - 1, 0, len(gy) - 1), np.clip(j + 1, 0, len(gy) - 1)
            deny = cost[e, jm, i] - 2 * cost[e, j, i] + cost[e, jp, i]
            m = (
                (j > 0)
                & (j < len(gy) - 1)
                & (deny > 0)
                & np.isfinite(cost[e, jm, i])
                & np.isfinite(cost[e, jp, i])
            )
            y[m] += (
                0.5
                * (gy[j[m] + 1] - gy[j[m]])
                * (cost[e[m], j[m] - 1, i[m]] - cost[e[m], j[m] + 1, i[m]])
                / deny[m]
            )
            result[sl, 0], result[sl, 1] = x, y
        return result
