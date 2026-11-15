"""Area analysis tests."""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("EARTHSCOPE_DATA", "sample")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from science.area import analyze_area, area_mask, get_grid, polygon_mask  # noqa: E402

# polygon mask: a 10° box holds 100 cells
lat = 90 - (np.arange(180) + 0.5)
lon = -180 + (np.arange(360) + 0.5)
box = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
assert polygon_mask(box, lat, lon).sum() == 100

# antimeridian
am = [[175, -10], [-175, -10], [-175, 0], [175, 0], [175, -10]]
assert polygon_mask(am, lat, lon).sum() == 100

# orientation does not matter
assert polygon_mask(box[::-1], lat, lon).sum() == 100

# point and tiny polygon
g = get_grid("t2m_c")
assert area_mask(g, point={"lat": 4.6, "lng": -74.1}).sum() == 1
tiny = {"type": "Polygon", "coordinates": [[[-74.1, 4.6], [-74.0, 4.6], [-74.0, 4.7], [-74.1, 4.6]]]}
assert area_mask(g, geometry=tiny).sum() == 1

# full analysis schema
r = analyze_area("t2m_c", point={"lat": 40.4, "lng": -3.7})
assert r["ok"], r
for key in ("monthly", "annual", "trend_line", "trend", "robustness", "caveats", "doi", "area"):
    assert key in r, key
assert r["trend"]["verdict"] in ("significant increasing trend", "significant decreasing trend",
                                 "no statistically significant trend")

# land-only variable over the ocean
r = analyze_area("evi", point={"lat": 0.0, "lng": -140.0})
assert not r["ok"] and "land only" in r["reason"], r

# errors
for bad in ({"type": "Point", "coordinates": [0, 0]},):
    try:
        analyze_area("t2m_c", geometry=bad)
        raise AssertionError("non-polygon geometry should fail")
    except ValueError:
        pass

print(f"OK - polygon, antimeridian, point and land-only checks pass ({g.source} data, {g.label} grid)")
