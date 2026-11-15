"""Dataset tests."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from science import dataset  # noqa: E402

m = dataset.manifest()
assert len(m["variables"]) >= 5, m["variables"].keys()
for var, v in m["variables"].items():
    files = sorted((dataset.DATASET / var).glob("*.i16.xz"))
    assert len(files) == len(v["years"]) == v["files"], var
    year = v["years"][len(v["years"]) // 2]
    a = dataset.load_year(var, year)
    assert a.shape == (12, 360, 720) and a.dtype == np.float32, (var, a.shape)
    finite = a[np.isfinite(a)]
    assert finite.size > 0, var

# round-trip of the codec on random data with gaps
rng = np.random.default_rng(0)
x = rng.normal(20, 10, (2, 360, 720)).astype("float32")
x[0, :10] = np.nan
y = dataset.decode(dataset.encode(x, 0.01), 2, 0.01)
assert (np.isnan(x) == np.isnan(y)).all() and np.nanmax(np.abs(x - y)) <= 0.0051
total = sum(v["bytes"] for v in m["variables"].values()) / 1e6
print(f"OK - {len(m['variables'])} variables, {sum(len(v['years']) for v in m['variables'].values())} "
      f"variable-years, {total:.0f} MB; codec round-trip within half a step")
