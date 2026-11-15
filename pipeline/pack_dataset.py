"""Pack the global grids into the compressed dataset in data/dataset/."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from science.dataset import DATASET, PACKING, decode, encode  # noqa: E402
from science.variables import GLOBAL_VARIABLES  # noqa: E402

RAW = ROOT / "data" / "global"

EXTRA = {  # stored but not shown on the web yet
    "ndvi": {"label": "Vegetation (NDVI)", "units": "index", "dataset": "MODIS/Terra MOD13C2 v6.1",
             "doi": "https://doi.org/10.5067/MODIS/MOD13C2.061", "kind": "satellite"},
    "lst_night_c": {"label": "Land surface temperature (night)", "units": "°C",
                    "dataset": "MODIS/Terra MOD11C3 v6.1", "doi": "https://doi.org/10.5067/MODIS/MOD11C3.061",
                    "kind": "satellite"},
    "qv2m_gkg": {"label": "Specific humidity (2 m)", "units": "g/kg",
                 "dataset": "MERRA-2 M2TMNXSLV v5.12.4", "doi": "https://doi.org/10.5067/AP1B0BA5PD2K",
                 "kind": "reanalysis"},
}


def pack(var: str, manifest: dict) -> None:
    product, scale = PACKING[var]
    files = sorted((RAW / product).glob("*.npz"))
    by_year: dict[int, list[Path]] = {}
    for f in files:
        by_year.setdefault(int(f.stem[:4]), []).append(f)
    years = [y for y, fs in sorted(by_year.items()) if len(fs) == 12]
    if not years:
        sys.exit(f"{var}: no complete years in {RAW / product} (run pipeline/fetch_global.py)")
    (DATASET / var).mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    total, max_err = 0, 0.0
    for y in years:
        arr = np.stack([np.load(f)[var] for f in by_year[y]]).astype("float32")
        blob = encode(arr, scale)
        back = decode(blob, 12, scale)
        ok = np.isnan(arr) == np.isnan(back)
        assert ok.all(), f"{var} {y}: NaN mask changed"
        err = float(np.nanmax(np.abs(arr - back))) if np.isfinite(arr).any() else 0.0
        assert err <= scale * 0.51 or np.nanmax(np.abs(arr)) > 32767 * scale, f"{var} {y}: error {err}"
        max_err = max(max_err, err)
        (DATASET / var / f"{y}.i16.xz").write_bytes(blob)
        h.update(blob)
        total += len(blob)
    meta = {**EXTRA.get(var, {}), **{k: GLOBAL_VARIABLES[var][k] for k in ("label", "units", "dataset", "doi", "kind")
                                    if var in GLOBAL_VARIABLES}}
    manifest["variables"][var] = {**meta, "product": product, "scale": scale, "years": years,
                                  "files": len(years), "bytes": total, "max_abs_error": round(max_err, 6),
                                  "sha": h.hexdigest()[:16]}
    print(f"{var}: {len(years)} years, {total / 1e6:.1f} MB, max error {max_err:.5f} (scale {scale})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", choices=list(PACKING), action="append")
    args = ap.parse_args()
    DATASET.mkdir(parents=True, exist_ok=True)
    mp = DATASET / "manifest.json"
    manifest = json.loads(mp.read_text()) if mp.exists() else {}
    manifest.update({
        "name": "EarthScope AI global monthly dataset",
        "grid": {"resolution_deg": 0.5, "nlat": 360, "nlon": 720, "order": "north→south, west→east",
                 "lat_first": 89.75, "lon_first": -179.75},
        "encoding": "int16 × scale (NaN = -32768) · delta along longitude · byte shuffle · LZMA (xz)",
        "reader": "science/dataset.py",
        "built": str(date.today()),
    })
    manifest.setdefault("variables", {})
    t0 = time.time()
    for v in args.var or PACKING:
        pack(v, manifest)
    mp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    size = sum(v["bytes"] for v in manifest["variables"].values())
    print(f"done in {time.time() - t0:.0f}s · dataset {size / 1e6:.0f} MB → {DATASET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
