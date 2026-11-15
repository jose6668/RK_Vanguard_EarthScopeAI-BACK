"""EarthScope AI analysis API (FastAPI)."""
from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# claves en .env, si existe
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from export.build_site_data import clean  # noqa: E402
from science.area import analyze_area, data_mode, get_grid  # noqa: E402
from science.variables import GLOBAL_VARIABLES  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

app = FastAPI(title="EarthScope AI API", version="1.0.0",
              description="Trend analysis of any area of the planet on NASA global grids (0.5°, 2001–2024).")
# public read-only data, no credentials
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class Point(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class Polygon(BaseModel):
    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class AreaRequest(BaseModel):
    variable: str
    geometry: Optional[Polygon] = None
    point: Optional[Point] = None
    alpha: float = Field(0.05, gt=0, lt=0.5)


@lru_cache(maxsize=256)
def _cached(var: str, geometry_json: str, point_json: str, alpha: float) -> dict:
    geometry = json.loads(geometry_json) if geometry_json else None
    point = json.loads(point_json) if point_json else None
    return clean(analyze_area(var, geometry=geometry, point=point, alpha=alpha))


@app.get("/api/v1/health")
def health():
    out = {}
    for k in GLOBAL_VARIABLES:
        try:
            g = get_grid(k)
            out[k] = {"source": g.source, "grid": g.label, "months": len(g.months),
                      "first": g.months[0], "last": g.months[-1]}
        except FileNotFoundError as exc:
            out[k] = {"source": None, "error": str(exc)}
    return {"status": "ok", "data_mode": data_mode(), "variables": out}


@app.get("/api/v1/variables")
def variables():
    return GLOBAL_VARIABLES


@app.get("/api/v1/forecast")
def forecast_info():
    """Años y variables que predice la red neuronal (vacío si no hay forecast.npz)."""
    from api import forecast

    return forecast.info()


@app.get("/api/v1/forecast/map/{var}/{year}.png")
def forecast_map(var: str, year: int, min: float, max: float):
    """Mapa anual predicho por la red, coloreado con la escala que manda la web."""
    from fastapi import Response

    from api import forecast

    if var not in forecast.COLORS or not max > min:
        raise HTTPException(422, "unknown variable or bad scale")
    try:
        png = forecast.map_png(var, year, round(min, 4), round(max, 4))
    except (FileNotFoundError, KeyError):
        raise HTTPException(404, "no forecast for that variable and year")
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/v1/forecast/point/{var}")
def forecast_point(var: str, lat: float, lng: float):
    """Predicción anual de la red en un punto, con la dispersión del ensemble."""
    from api import forecast

    try:
        return {"variable": var, "annual": forecast.point(var, lat, lng)}
    except (FileNotFoundError, KeyError):
        raise HTTPException(404, "no forecast for that variable")


@app.post("/api/v1/area/analyze")
def area_analyze(req: AreaRequest):
    if req.variable not in GLOBAL_VARIABLES:
        raise HTTPException(404, f"unknown variable '{req.variable}'")
    if not req.geometry and not req.point:
        raise HTTPException(422, "send a GeoJSON 'geometry' (Polygon) or a 'point'")
    try:
        return _cached(req.variable,
                       json.dumps(req.geometry.model_dump(), sort_keys=True) if req.geometry else "",
                       json.dumps(req.point.model_dump(), sort_keys=True) if req.point else "",
                       req.alpha)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


# satélites: TLE de CelesTrak con caché de 6 horas
SATS = [25544, 25994, 27424, 39574, 37849, 43013, 49260, 40376, 43613]
_TLE_CACHE = ROOT / "data" / ".cache" / "tle.json"
_tle: dict = {}


def _fetch_tles() -> list[dict]:
    import httpx
    out = []
    with httpx.Client(timeout=10, follow_redirects=True) as cl:
        for n in SATS:
            try:
                r = cl.get("https://celestrak.org/NORAD/elements/gp.php", params={"CATNR": n, "FORMAT": "TLE"})
                lines = [x.strip() for x in r.text.strip().splitlines() if x.strip()]
                if r.status_code == 200 and len(lines) >= 3 and lines[1].startswith("1 ") and lines[2].startswith("2 "):
                    out.append({"norad": n, "name": lines[0], "line1": lines[1], "line2": lines[2]})
            except httpx.HTTPError:
                continue
    return out


@app.get("/api/v1/satellites")
def satellites():
    import time
    now = time.time()
    if _tle.get("t", 0) > now - 6 * 3600:
        return _tle["data"]
    sats = _fetch_tles()
    if sats:
        data = {"source": "celestrak", "fetched": now, "satellites": sats}
        _tle.update(t=now, data=data)
        try:
            _TLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _TLE_CACHE.write_text(json.dumps(data))
        except OSError:
            pass
        return data
    # sin red: última copia guardada, si existe
    if _TLE_CACHE.exists():
        return json.loads(_TLE_CACHE.read_text())
    raise HTTPException(503, "satellite data unavailable (no connection to CelesTrak)")


@app.get("/api/v1/events")
def events():
    """Eventos naturales activos (NASA EONET), con caché de 30 minutos."""
    from api import events as ev

    try:
        return ev.get()
    except RuntimeError:
        raise HTTPException(503, "natural events unavailable (no connection to NASA EONET)")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session: str = Field(min_length=1, max_length=100)
    context: Optional[dict] = None


@app.post("/api/v1/agent/chat")
def agent_chat(req: ChatRequest):
    try:
        from api import agent
    except ImportError:
        raise HTTPException(503, "agent not installed (pip install langchain langchain-anthropic)")
    if not agent.available():
        raise HTTPException(503, f"agent not configured (no API key for {agent.MODEL})")
    try:
        return agent.ask(req.message, req.session, req.context)
    except Exception as exc:  # noqa: BLE001
        # error del proveedor del modelo
        raise HTTPException(502, f"agent error: {exc}")


@app.post("/api/v1/agent/voice")
async def agent_voice(request: Request, session: str = "voz", context: str = ""):
    """Audio (WAV) en el cuerpo: transcribe y pregunta al agente. Devuelve transcript, reply y actions."""
    try:
        from api import agent
    except ImportError:
        raise HTTPException(503, "agent not installed (pip install langchain langchain-anthropic)")
    if not agent.stt_available():
        raise HTTPException(503, "voice not configured (set ELEVENLABS_API_KEY or GOOGLE_API_KEY)")
    audio = await request.body()
    if not audio or len(audio) > 5_000_000:
        raise HTTPException(422, "send 0-5 MB of audio in the body")
    mime = request.headers.get("content-type", "audio/wav").split(";")[0]
    try:
        ctx = json.loads(context) if context else None
    except json.JSONDecodeError:
        ctx = None
    try:
        text = agent.transcribe(audio, mime)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"transcription error: {exc}")
    if not text:
        return {"transcript": "", "reply": "", "actions": []}
    if not agent.available():
        raise HTTPException(503, f"agent not configured (no API key for {agent.MODEL})")
    try:
        return {"transcript": text, **agent.ask(text, session, ctx)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"agent error: {exc}")


@app.on_event("startup")
def warm():
    """Build/open the memory-mapped cubes so the first request is fast."""
    for k in GLOBAL_VARIABLES:
        try:
            get_grid(k)
        except FileNotFoundError:
            pass
