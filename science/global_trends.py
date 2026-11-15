"""Per-pixel trend maps for global grids (vectorised).

For each grid cell, a Mann-Kendall test and a Sen slope on its ANNUAL means (one value per
year, so the seasonal cycle is already removed). Used to build global "where is it changing"
maps. Regional analyses keep using the stronger seasonal test in science/trends.py.

Caveats (shown in the UI):
- Annual series of ~24 values: low power; p-values ignore residual autocorrelation.
- Each pixel is tested independently: at α = 0.05, ~5 % of pixels look significant by chance.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm


def annual_trend(cube: np.ndarray, years: np.ndarray, min_years: int = 15) -> dict[str, np.ndarray]:
    """cube: (n_years, lat, lon) of annual means (NaN = missing).

    Returns maps: sen_slope (units/yr), p_value, n (valid years), mean.
    """
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


def _angle(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle angle between two points, degrees (1° ≈ 111 km)."""
    r = np.deg2rad
    c = (np.sin(r(lat1)) * np.sin(r(lat2)) + np.cos(r(lat1)) * np.cos(r(lat2)) * np.cos(r(lon1 - lon2)))
    return float(np.rad2deg(np.arccos(np.clip(c, -1, 1))))


def hotspots(per_decade: np.ndarray, significant: np.ndarray, lat: np.ndarray, lon: np.ndarray,
             block: int = 10, n: int = 6, min_sep_deg: float = 15.0) -> list[dict]:
    """Where the planet changed the most: blocks of `block` x `block` cells (5° at 0.5°) where at
    least half of the cells have a significant trend, ranked by the mean change of those cells.
    Picks up to `n` places at least `min_sep_deg` apart along the sphere (≈ 1 650 km for 15°). Values are in units per decade.

    (Moved here from the web, which approximated it by decoding map colours.)
    """
    H, W = per_decade.shape
    cands = []
    for by in range(0, H, block):
        for bx in range(0, W, block):
            v = per_decade[by:by + block, bx:bx + block]
            s = significant[by:by + block, bx:bx + block] & np.isfinite(v)
            if s.sum() < (block * block) / 2:
                continue
            cands.append((abs(float(v[s].mean())), float(v[s].mean()),
                          float(lat[min(by + block // 2, H - 1)]), float(lon[min(bx + block // 2, W - 1)]),
                          int(s.sum())))
    cands.sort(reverse=True)
    out: list[dict] = []
    for _, val, la, lo, cells in cands:
        if any(_angle(o["lat"], o["lng"], la, lo) < min_sep_deg for o in out):   # great-circle distance
            continue
        out.append({"lat": round(la, 2), "lng": round(lo, 2), "value_per_decade": val, "significant_cells": cells})
        if len(out) >= n:
            break
    return out
