"""The EarthScope global dataset as it lives IN GIT (data/dataset/).

Monthly global grids 2001–2024 at 0.5° (360 x 720, north→south, west→east) for every variable
downloaded by pipeline/fetch_global.py, packed so that cloning the repo is enough:

  data/dataset/manifest.json            variables, scale factors, years, grid, provenance
  data/dataset/<var>/<YYYY>.i16.xz      12 months of one variable

Encoding (lossless at the stated precision, like NASA's own scale_factor packing):
  value ≈ int16 × scale        (NaN → −32768)
  delta along longitude → byte-shuffle (high/low bytes) → LZMA (xz)
Precision: 0.01 mm/day, 0.0001 index, 0.01 °C, 0.001 g/kg, 0.01 %.

Build:  python pipeline/pack_dataset.py      Read:  load_cube("t2m_c")
"""
from __future__ import annotations

import json
import lzma
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset"
CACHE = ROOT / "data" / ".cache"
NLAT, NLON = 360, 720
NODATA = -32768

# variable -> (product folder in data/global, scale factor)
PACKING = {
    "precip_mm_day": ("precipitation", 0.01),
    "ndvi": ("vegetation", 0.0001),
    "evi": ("vegetation", 0.0001),
    "lst_day_c": ("lst", 0.01),
    "lst_night_c": ("lst", 0.01),
    "t2m_c": ("atmosphere", 0.01),
    "qv2m_gkg": ("atmosphere", 0.001),
    "rh2m_pct": ("atmosphere", 0.01),
}


# ------------------------------------------------------------------ codec
def encode(values: np.ndarray, scale: float) -> bytes:
    """(months, 360, 720) float → compressed bytes."""
    q = np.where(np.isnan(values), NODATA, np.round(values / scale))
    q = np.clip(q, -32767, 32767).astype("<i2")
    q[np.isnan(values)] = NODATA
    d = np.diff(q, axis=-1, prepend=np.zeros(q.shape[:-1] + (1,), "<i2")).astype("<i2")  # wraps like int16
    shuffled = np.frombuffer(d.tobytes(), np.uint8).reshape(-1, 2).T.copy()
    return lzma.compress(shuffled.tobytes(), preset=9)


def decode(blob: bytes, months: int, scale: float) -> np.ndarray:
    """Compressed bytes → (months, 360, 720) float32 with NaN for missing data."""
    shuffled = np.frombuffer(lzma.decompress(blob), np.uint8).reshape(2, -1)
    d = shuffled.T.copy().view("<i2").reshape(months, NLAT, NLON)
    q = np.cumsum(d, axis=-1, dtype="<i2")                                  # int16 wrap undoes diff
    out = q.astype("float32") * np.float32(scale)
    out[q == NODATA] = np.nan
    return out


# ------------------------------------------------------------------ reading
@lru_cache(maxsize=1)
def manifest() -> dict:
    p = DATASET / "manifest.json"
    if not p.exists():
        raise FileNotFoundError("data/dataset/ not found: clone the full repo or run pipeline/pack_dataset.py")
    return json.loads(p.read_text())


def available() -> bool:
    return (DATASET / "manifest.json").exists()


def months(var: str) -> list[str]:
    v = manifest()["variables"][var]
    return [f"{y}-{m:02d}" for y in v["years"] for m in range(1, 13)]


def load_year(var: str, year: int) -> np.ndarray:
    v = manifest()["variables"][var]
    return decode((DATASET / var / f"{year}.i16.xz").read_bytes(), 12, v["scale"])


def load_cube(var: str, use_cache: bool = True) -> tuple[np.ndarray, list[str]]:
    """All months of one variable as float32 (months, 360, 720).

    With use_cache, the decoded cube is kept as a memory-mapped file in data/.cache/ (out of git)
    so the API starts fast after the first run.
    """
    ms = months(var)
    if not use_cache:
        return np.concatenate([load_year(var, y) for y in manifest()["variables"][var]["years"]]), ms
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = manifest()["variables"][var]["sha"]
    cube_p, meta_p = CACHE / f"{var}.npy", CACHE / f"{var}.json"
    if not (cube_p.exists() and meta_p.exists() and json.loads(meta_p.read_text()).get("sha") == stamp):
        cube = np.lib.format.open_memmap(cube_p, mode="w+", dtype="float32", shape=(len(ms), NLAT, NLON))
        for i, y in enumerate(manifest()["variables"][var]["years"]):
            cube[i * 12:(i + 1) * 12] = load_year(var, y)
        cube.flush()
        meta_p.write_text(json.dumps({"sha": stamp}))
    return np.load(cube_p, mmap_mode="r"), ms


def grid() -> tuple[np.ndarray, np.ndarray]:
    lat = 90 - (np.arange(NLAT) + 0.5) * 0.5
    lon = -180 + (np.arange(NLON) + 0.5) * 0.5
    return lat, lon
