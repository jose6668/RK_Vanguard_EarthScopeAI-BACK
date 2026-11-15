"""Per-pixel trend maps for global grids."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm


def annual_trend(cube: np.ndarray, years: np.ndarray, min_years: int = 15) -> dict[str, np.ndarray]:
    """cube: (n_years, lat, lon) of annual means (NaN = missing)."""
    n_t = cube.shape[0]
    x = cube.reshape(n_t, -1).astype("float64")
    valid = ~np.isnan(x)
    n = valid.sum(0)

    s = np.zeros(x.shape[1])
    slopes = []
    for i in range(n_t - 1):
        d = x[i + 1:] - x[i]                                    # (n_t-i-1, pix)
        s += np.nansum(np.sign(d), axis=0)
        slopes.append(d / (years[i + 1:] - years[i])[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sen = np.nanmedian(np.concatenate(slopes, axis=0), axis=0)
        mean = np.nanmean(x, axis=0)

    var = n * (n - 1) * (2 * n + 5) / 18.0                      # no-ties variance
    z = np.where(s > 0, (s - 1) / np.sqrt(np.maximum(var, 1e-12)),
                 np.where(s < 0, (s + 1) / np.sqrt(np.maximum(var, 1e-12)), 0.0))
    p = 2 * (1 - norm.cdf(np.abs(z)))

    bad = n < min_years
    for a in (sen, p, mean):
        a[bad] = np.nan
    shape = cube.shape[1:]
    return {"sen_slope": sen.reshape(shape), "p_value": p.reshape(shape),
            "n": n.reshape(shape), "mean": mean.reshape(shape)}


def area_weighted_mean(grid: np.ndarray, lat: np.ndarray, mask: np.ndarray | None = None) -> float:
    """cos(lat)-weighted mean over valid cells (optionally restricted to a fixed mask)."""
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(grid)
    ok = ~np.isnan(grid) if mask is None else (mask & ~np.isnan(grid))
    return float(np.sum(grid[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
