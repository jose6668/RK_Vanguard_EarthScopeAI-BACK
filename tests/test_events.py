"""Eventos EONET: parseo y endpoint sin red."""
import os
import sys
from pathlib import Path

os.environ.setdefault("EARTHSCOPE_DATA", "sample")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

import api.events as ev  # noqa: E402
from api.main import app  # noqa: E402

raw = {"events": [
    {"id": "EONET_1", "title": "Wildfire in Bolivia", "categories": [{"id": "wildfires", "title": "Wildfires"}],
     "sources": [{"url": "https://firms.example"}],
     "geometry": [{"type": "Point", "date": "2026-11-10T00:00:00Z", "coordinates": [-63.5, -17.2],
                   "magnitudeValue": 1200, "magnitudeUnit": "acres"}]},
    {"id": "EONET_2", "title": "Hurricane Test", "categories": [{"id": "severeStorms", "title": "Severe Storms"}],
     "geometry": [{"type": "Point", "date": "2026-11-12T00:00:00Z", "coordinates": [-70, 15]},
                  {"type": "Point", "date": "2026-11-13T00:00:00Z", "coordinates": [-72, 17]}]},
    {"id": "EONET_3", "title": "Polygon only", "categories": [{"id": "floods"}], "geometry": [{"type": "Polygon"}]},
]}
ev_list = ev.parse(raw)
assert [e["id"] for e in ev_list] == ["EONET_2", "EONET_1"], ev_list
assert ev_list[0]["kind"] == "storm" and ev_list[0]["lat"] == 17 and ev_list[0]["track"] == [[15, -70], [17, -72]]
assert ev_list[1]["kind"] == "fire" and ev_list[1]["magnitude"] == "1200 acres"

c = TestClient(app)
ev.CACHE = Path("/tmp/es_eonet_test.json")
ev.CACHE.unlink(missing_ok=True)
ev._mem.clear()


def _boom():
    raise OSError("offline")


ev.fetch = _boom
assert c.get("/api/v1/events").status_code == 503
ev.fetch = lambda: ev_list
assert c.get("/api/v1/events").json()["events"][0]["id"] == "EONET_2"
ev._mem.clear()
ev.fetch = _boom
assert len(c.get("/api/v1/events").json()["events"]) == 2  # copia guardada
print("test_events: OK")
