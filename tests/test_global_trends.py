"""Validate per-pixel trend maps on synthetic grids with KNOWN trends (python tests/test_global_trends.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from science.global_trends import annual_trend, hotspots  # noqa: E402

rng = np.random.default_rng(0)
years = np.arange(2001, 2025)
cube = rng.normal(0, 1, (24, 40, 50))
cube[:, :20, :] += (0.15 * (years - 2001))[:, None, None]      # trend of +0.15/yr in the top half
cube[3, 0, 0] = np.nan                                           # one gap
cube[:, 39, 49] = np.nan                                         # empty pixel

r = annual_trend(cube, years)
slope = float(np.nanmedian(r["sen_slope"][:20]))
power = float(np.mean(r["p_value"][:20] < 0.05))
false_pos = float(np.nanmean(r["p_value"][20:] < 0.05))
assert abs(slope - 0.15) < 0.02, slope
assert power > 0.95, power
assert false_pos < 0.08, false_pos
assert np.isnan(r["sen_slope"][39, 49]) and r["n"][0, 0] == 23
print(f"OK - slope 0.15 recovered as {slope:.3f}; power {power:.0%}; false positives {false_pos:.1%}")

# hotspots: the strongest significant block comes first, weak or non-significant ones are skipped
lat = 90 - (np.arange(40) + 0.5) * 0.5
lon = -180 + (np.arange(50) + 0.5) * 0.5
per = np.zeros((40, 50)); sig = np.zeros((40, 50), bool)
per[0:10, 0:10], sig[0:10, 0:10] = 2.0, True          # strong
per[20:30, 30:40], sig[20:30, 30:40] = -1.0, True      # weaker, opposite sign
per[30:40, 0:10] = 5.0                                  # strong but NOT significant
hs = hotspots(per, sig, lat, lon, min_sep_deg=3)
assert [round(h["value_per_decade"], 1) for h in hs] == [2.0, -1.0], hs
print("OK - hotspots rank significant blocks only")

