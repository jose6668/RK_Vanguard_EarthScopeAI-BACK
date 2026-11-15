"""Mapas de la predicción de la red: años disponibles, PNG con transparencia por incertidumbre."""
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
tmp = Path(tempfile.mkdtemp()) / "forecast.npz"
os.environ["EARTHSCOPE_FORECAST"] = str(tmp)
os.environ.setdefault("EARTHSCOPE_DATA", "sample")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from api.main import app  # noqa: E402

c = TestClient(app)
off = c.get("/api/v1/forecast").json()
assert off["available"] is False and off["status"] == "experimental" and off["caveat"]
assert c.get("/api/v1/forecast/map/evi/2026.png?min=0&max=1").status_code == 404

months = [f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)] + ["2027-01"]
data = np.random.rand(len(months), 2, 36, 72).astype(np.float32)
data[:, :, :5] = np.nan
spread = np.zeros_like(data)
spread[:, :, 20:] = 0.5  # sur muy incierto
np.savez_compressed(tmp, data=data, spread=spread, months=np.array(months), vars=np.array(["evi", "t2m_c"]))

info = c.get("/api/v1/forecast").json()
assert info["status"] == "experimental" and "Experimental" in info["caveat"] and info["validated_horizon_months"] == 1
assert info["available"] and info["years"] == [2025, 2026] and info["variables"] == ["evi", "t2m_c"], info
r = c.get("/api/v1/forecast/map/evi/2026.png?min=0&max=1")
assert r.status_code == 200 and r.headers["content-type"] == "image/png"
a = np.array(Image.open(io.BytesIO(r.content)))
assert a.shape == (360, 720, 4)
assert a[10, 100, 3] == 0  # sin dato
assert a[100, 100, 3] > a[300, 100, 3] > 0  # más incierto, más transparente
assert c.get("/api/v1/forecast/map/precip_mm_day/2026.png?min=0&max=8").status_code == 404
assert c.get("/api/v1/forecast/map/evi/2026.png?min=1&max=0").status_code == 422
p = c.get("/api/v1/forecast/point/evi?lat=-30&lng=0").json()["annual"]
assert [x["year"] for x in p] == [2025, 2026] and p[0]["spread"] == 0.5, p
assert c.get("/api/v1/forecast/point/evi?lat=89&lng=0").json()["annual"] == []  # sin dato
assert c.get("/api/v1/forecast/point/precip_mm_day?lat=0&lng=0").status_code == 404
print("test_forecast: OK")
