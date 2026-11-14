"""Build the lightweight SAMPLE of the global data that lives in git (data/sample/).

The full dataset (data/dataset/, 0.5°, ≈ 370 MB compressed) is in git but heavy to load. This sample keeps every
month (2001–2024) and every variable but coarsens the grid to 2° (90 x 180) and stores float16,
so teammates can run and edit the API and science code without downloading NASA data.

Results computed on the sample are for DEVELOPMENT only (the API labels them "sample").

    python pipeline/make_sample.py        # from the packed dataset (data/dataset/)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from science import dataset  # noqa: E402
from science.variables import GLOBAL_VARIABLES  # noqa: E402

OUT = ROOT / "data" / "sample"
FACTOR = 4                                   # 0.5° -> 2°
RES = 0.5 * FACTOR


def coarsen(a: np.ndarray) -> np.ndarray:
    t, r, c = a.shape
    b = a.reshape(t, r // FACTOR, FACTOR, c // FACTOR, FACTOR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(2, 4))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lat = (90 - (np.arange(int(180 / RES)) + 0.5) * RES).astype("float32")
    lon = (-180 + (np.arange(int(360 / RES)) + 0.5) * RES).astype("float32")
    for var, meta in GLOBAL_VARIABLES.items():
        cube, months = dataset.load_cube(var, use_cache=False)
        small = coarsen(np.asarray(cube, dtype="float32")).astype("float16")
        np.savez_compressed(OUT / f"{var}.npz", values=small, lat=lat, lon=lon, months=np.array(months))
        print(f"{var}: {small.shape} → {(OUT / f'{var}.npz').stat().st_size / 1e6:.1f} MB")
    (OUT / "README.md").write_text(
        "# Muestra de datos globales (2°, float16)\n\n"
        "Generada con `python pipeline/make_sample.py` a partir de `data/dataset/` (0,5°).\n"
        "Sirve para desarrollar y probar la API y `science/` sin descargar datos de NASA.\n"
        "**No usar sus resultados como conclusiones**: la API los marca como `sample`.\n")
    (OUT / "manifest.json").write_text(json.dumps({"resolution_deg": RES, "dtype": "float16",
                                                   "source": "data/dataset (0.5°)",
                                                   "variables": list(GLOBAL_VARIABLES)}, indent=1))


if __name__ == "__main__":
    main()
