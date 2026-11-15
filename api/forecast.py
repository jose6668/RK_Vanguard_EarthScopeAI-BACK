"""Predicción de la red neuronal (forecast.npz) como mapas anuales para el globo.

IMPORTANTE (reglas del proyecto): es una proyección EXPERIMENTAL. model/train_nn.py valida la red
solo a 1 mes vista (test 2023-2024 contra persistencia y climatología). Los meses siguientes son un
encadenamiento autorregresivo sin validación por horizonte y sin escenario de emisiones, así que
tienden a la climatología 2001-2020. No es una proyección climática. Toda respuesta lo declara.
"""
from __future__ import annotations

import io
import warnings
from functools import lru_cache

import numpy as np
from PIL import Image

from export.build_site_data import colorize

STATUS = "experimental"
CAVEAT = ("Experimental neural-network projection. Validated only one month ahead (2023-2024 test); "
          "later months are an unvalidated autoregressive rollout with no emissions scenario that drifts "
          "toward the 2001-2020 climatology. Not a climate projection or a forecast.")

# mismas paletas que los mapas históricos (export/build_global_data.py)
COLORS = {
    "precip_mm_day": ["#f1f5f9", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e1b4b"],
    "evi": ["#a16207", "#d9b36c", "#86efac", "#22c55e", "#14532d"],
    "lst_day_c": ["#1e3a8a", "#38bdf8", "#fde047", "#f97316", "#7f1d1d"],
    "t2m_c": ["#1e3a8a", "#38bdf8", "#fde047", "#f97316", "#7f1d1d"],
    "rh2m_pct": ["#7c2d12", "#f59e0b", "#fef3c7", "#5eead4", "#0f766e"],
}


def _load():
    from api.agent import FORECAST

    if not FORECAST.exists():
        return None
    return _read(str(FORECAST), FORECAST.stat().st_mtime)


@lru_cache(maxsize=1)
def _read(path: str, _mtime: float):
    f = np.load(path)
    months = [str(m) for m in f["months"]]
    return f["data"], (f["spread"] if "spread" in f.files else None), months, [str(v) for v in f["vars"]]


def _report() -> dict | None:
    """Métricas de test de train_nn.py (report.json junto a forecast.npz), si se copiaron."""
    import json

    from api.agent import FORECAST

    for p in (FORECAST.with_name("report.json"), FORECAST.with_name("forecast_report.json")):
        if p.exists():
            try:
                return json.loads(p.read_text()).get("test")
            except (OSError, ValueError):
                return None
    return None


def info() -> dict:
    fc = _load()
    if fc is None:
        return {"available": False, "years": [], "variables": [], "status": STATUS, "caveat": CAVEAT}
    _, _, months, vars_ = fc
    years = sorted({int(m[:4]) for m in months})
    # solo años con al menos 10 meses predichos
    years = [y for y in years if sum(m.startswith(str(y)) for m in months) >= 10]
    return {"available": True, "years": years, "variables": [v for v in vars_ if v in COLORS],
            "first": months[0], "last": months[-1],
            "status": STATUS, "caveat": CAVEAT, "validated_horizon_months": 1, "test_skill": _report()}


def annual(var: str, year: int) -> tuple[np.ndarray, np.ndarray | None]:
    fc = _load()
    if fc is None:
        raise FileNotFoundError("no forecast")
    data, spread, months, vars_ = fc
    if var not in vars_:
        raise KeyError(var)
    idx = [i for i, m in enumerate(months) if m.startswith(str(year))]
    if not idx:
        raise KeyError(year)
    v = vars_.index(var)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(data[idx, v], axis=0)
        sp = np.nanmean(spread[idx, v], axis=0) if spread is not None else None
    return mean, sp


@lru_cache(maxsize=256)
def map_png(var: str, year: int, vmin: float, vmax: float) -> bytes:
    """Mapa anual predicho con la escala de la web. Donde el ensemble duda más, el color se desvanece."""
    mean, sp = annual(var, year)
    img = np.array(colorize(mean, vmin, vmax, COLORS[var], alpha_ramp=var == "precip_mm_day"))
    if sp is not None:
        rel = np.nan_to_num(sp / max(vmax - vmin, 1e-9), nan=0)
        fade = np.clip(1 - rel * 4, 0.35, 1)
        img[..., 3] = (img[..., 3] * fade).astype(np.uint8)
    out = Image.fromarray(img).resize((720, 360), Image.NEAREST)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def point(var: str, lat: float, lng: float) -> list[dict]:
    """Promedio anual predicho (y su dispersión) en un punto."""
    fc = _load()
    if fc is None:
        raise FileNotFoundError("no forecast")
    data, spread, months, vars_ = fc
    if var not in vars_:
        raise KeyError(var)
    v = vars_.index(var)
    h, w = data.shape[-2:]
    y = min(max(int((90 - lat) / 180 * h), 0), h - 1)
    x = min(max(int((lng + 180) / 360 * w), 0), w - 1)
    out = []
    for yr in info()["years"]:
        idx = [i for i, m in enumerate(months) if m.startswith(str(yr))]
        vals = data[idx, v, y, x]
        if not np.isfinite(vals).any():
            continue
        sp = spread[idx, v, y, x] if spread is not None else None
        out.append({"year": yr, "value": round(float(np.nanmean(vals)), 4),
                    "spread": round(float(np.nanmean(sp)), 4) if sp is not None else None})
    return out
