"""Satélites: endpoint con TLE simulados, sin red."""
import os
import sys
from pathlib import Path

os.environ.setdefault("EARTHSCOPE_DATA", "sample")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

import api.main as m  # noqa: E402

c = TestClient(m.app)
m._tle.clear()
m._TLE_CACHE = Path("/tmp/es_tle_test.json")
m._TLE_CACHE.unlink(missing_ok=True)

m._fetch_tles = lambda: []
assert c.get("/api/v1/satellites").status_code == 503

fake = [{"norad": 25544, "name": "ISS (ZARYA)", "line1": "1 25544U ...", "line2": "2 25544 ..."}]
m._fetch_tles = lambda: fake
r = c.get("/api/v1/satellites").json()
assert r["source"] == "celestrak" and r["satellites"][0]["norad"] == 25544, r

# sin red pero con copia guardada
m._tle.clear()
m._fetch_tles = lambda: []
assert c.get("/api/v1/satellites").json()["satellites"] == fake
print("test_satellites: OK")
