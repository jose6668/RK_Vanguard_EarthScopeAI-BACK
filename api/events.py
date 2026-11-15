"""Eventos naturales activos de NASA EONET (incendios, tormentas, volcanes, inundaciones...)."""
from __future__ import annotations

import json
import time
from pathlib import Path

URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
CACHE = Path(__file__).resolve().parents[1] / "data" / ".cache" / "eonet.json"
TTL = 30 * 60
_mem: dict = {}

# categorías de EONET -> clave corta para la web
KINDS = {
    "wildfires": "fire", "severeStorms": "storm", "volcanoes": "volcano", "floods": "flood",
    "seaLakeIce": "ice", "earthquakes": "quake", "drought": "drought", "dustHaze": "dust",
    "landslides": "landslide", "snow": "snow", "tempExtremes": "heat", "waterColor": "water",
    "manmade": "manmade",
}


def fetch() -> list[dict]:
    import httpx

    r = httpx.get(URL, params={"status": "open", "days": 60, "limit": 400}, timeout=20)
    r.raise_for_status()
    return parse(r.json())


def parse(raw: dict) -> list[dict]:
    out = []
    for e in raw.get("events", []):
        geo = [g for g in e.get("geometry", []) if g.get("type") == "Point"]
        if not geo:
            continue
        last = geo[-1]
        lng, lat = last["coordinates"][:2]
        cat = (e.get("categories") or [{}])[0].get("id", "")
        item = {"id": e["id"], "title": e.get("title", ""), "kind": KINDS.get(cat, "other"),
                "category": (e.get("categories") or [{}])[0].get("title", ""),
                "lat": round(float(lat), 3), "lng": round(float(lng), 3), "date": last.get("date", "")[:10],
                "link": (e.get("sources") or [{}])[0].get("url") or e.get("link", "")}
        if last.get("magnitudeValue") is not None:
            item["magnitude"] = f'{last["magnitudeValue"]} {last.get("magnitudeUnit") or ""}'.strip()
        # trayectoria de tormentas
        if len(geo) > 1 and item["kind"] == "storm":
            item["track"] = [[round(g["coordinates"][1], 2), round(g["coordinates"][0], 2)] for g in geo[-40:]]
        out.append(item)
    out.sort(key=lambda x: x["date"], reverse=True)
    return out


def get() -> dict:
    now = time.time()
    if _mem.get("t", 0) > now - TTL:
        return _mem["data"]
    try:
        events = fetch()
    except Exception:  # noqa: BLE001
        events = []
    if events:
        data = {"source": "NASA EONET", "fetched": now, "events": events}
        _mem.update(t=now, data=data)
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(data))
        except OSError:
            pass
        return data
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    raise RuntimeError("EONET unavailable")
