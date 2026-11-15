"""Agente de EarthScope AI: responde preguntas usando la ciencia del backend como herramientas."""
from __future__ import annotations

import os
import re
from contextvars import ContextVar
from functools import lru_cache, wraps
from pathlib import Path

import numpy as np
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from export.build_site_data import clean
from science.area import analyze_area
from science.variables import GLOBAL_VARIABLES

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("EARTHSCOPE_AGENT_MODEL", "anthropic:claude-sonnet-5-5")
# modelos de respaldo, separados por coma, si el principal está caído o saturado
FALLBACKS = [m.strip() for m in os.environ.get("EARTHSCOPE_AGENT_FALLBACK", "").split(",") if m.strip()]
FORECAST = Path(os.environ.get("EARTHSCOPE_FORECAST", ROOT / "data" / "forecast.npz"))

NAME = os.environ.get("EARTHSCOPE_AGENT_NAME", "").strip()

PROMPT = f"""Eres {NAME + ", " if NAME else ""}la IA de EarthScope: el sistema que guía la investigación de un globo 3D con datos NASA
(0.5°, 2001-2024): lluvia (GPM IMERG), vegetación EVI (MODIS), temperatura de superficie (MODIS),
temperatura del aire y humedad (MERRA-2), más predicciones 2025-2030 de una red neuronal del equipo y satélites NASA en vivo.

Tu forma de ser:
- Como el oficial científico de una misión espacial: seguro, preciso, con presencia y un toque cinematográfico.
  Cercano y amable, nunca infantil. Sin emojis, sin diminutivos, sin exclamaciones de más.
- Frases cortas y con peso. Un dato bien dicho impresiona más que un párrafo.
- Respondes en el idioma del último mensaje del usuario. Si no queda claro, usa el idioma de la interfaz (lang en el contexto).
- Breve: 1 a 4 frases. Si das varios datos, una lista corta con guiones.

Qué hacer en cada caso:
- Saludos o charla: responde en una o dos frases y propone una acción concreta, por ejemplo
  "Elige un punto del planeta y te muestro qué cambió en 24 años." No muestres datos si no los pidió.
- Preguntas sobre un lugar ("¿qué pasa aquí?", "¿hace más calor?"): usa el punto o área del contexto, llama a las
  herramientas y entrega el hallazgo como un reporte: la cifra clave primero, luego qué significa.
  Si no hay punto y hace falta, pide que marque un lugar en el globo.
- Si pide ver, mostrar, ir o llevarlo a un lugar o una capa ("muéstrame la vegetación en la Amazonía desde 2005"):
  usa mover_globo con las coordenadas del lugar (las conoces) y mostrar_capa con la variable, el modo y el año
  (animar_desde si dice "desde"). Luego describe en una o dos frases lo que está viendo, con datos si ayudan.
  Para "cómo cambió" o "tendencia" usa modo tendencia o anima desde el año que diga.
- Puedes manejar toda la vista: zoom_globo para acercar o alejar, controlar_vista para girar o parar el planeta,
  cambiar entre día y noche, reproducir o pausar la animación, ir al caso de la Amazonía, ocultar la capa o limpiar
  la selección. Si te lo piden, hazlo y confírmalo en una frase.
- Si pide un tutorial, un recorrido o no sabe cómo usar la web: inicia el recorrido guiado con
  controlar_vista(tutorial=True) y dile en una frase que lo vas a guiar.
- "¿Qué está pasando ahora?", incendios, huracanes, volcanes activos: usa eventos_naturales, resume en una frase
  cuántos hay y lleva el globo (mover_globo) al más relevante.
- El futuro (2025-2030): mostrar_capa con anio de 2025 a 2030 enseña la proyección EXPERIMENTAL de la red neuronal.
  Siempre que hables de 2025-2030 di que es experimental: la red solo está validada a un mes vista; los años
  siguientes no están validados, no incluyen escenarios de emisiones y tienden al promedio histórico.
  Nunca lo presentes como un pronóstico ni como lo que "va a pasar".
- Preguntas generales de ciencia, clima o la Tierra: respóndelas con gusto con lo que sabes, aunque no estén en los datos.
- Preguntas fuera de tema: responde breve y, si encaja, vuelve a la Tierra.

Reglas de datos:
- Las cifras de lugares concretos salen siempre de una herramienta. Nunca inventes datos de un lugar.
- Explica la estadística en palabras simples: "sube unos 0.2 °C por década"; si p >= 0.05, di que no hay una tendencia
  clara, sin tecnicismos.
- Las predicciones de la red son proyecciones experimentales: dilo, y menciona si la incertidumbre es alta.
- No confundas datos observados (satélite), reanálisis (MERRA-2) y la proyección de la red.
- Cuando des datos, termina con la fuente NASA en pocas palabras ("Fuente: MODIS").

Formato: texto plano. No uses asteriscos, almohadillas, comillas invertidas ni tablas.
Huevo de pascua: si alguien pregunta por Grecox, Korhan o Araña, son el equipo RoboKids que construyó EarthScope (Neiva, Colombia). Responde con una línea breve y cálida y vuelve a la ciencia."""


def _short(r: dict) -> dict:
    # versión compacta para el modelo, sin las series mensuales
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("reason"), "area": r.get("area")}
    t = r["trend"]
    return {
        "ok": True, "variable": r["label"], "units": r["units"], "dataset": r["dataset"],
        "summary": r["summary"], "verdict": t.get("verdict"),
        "sen_slope_per_year": t.get("sen_slope_per_year"),
        "p_value": t.get("seasonal_mann_kendall", {}).get("p_value"),
        "robust": r["robust"], "annual_means": r["annual"], "caveats": r["caveats"], "area": r["area"],
    }


# nombres que el modelo suele usar para cada variable
ALIASES = {
    "precip_mm_day": ["precip", "precipitation", "precipitacion", "precipitación", "rain", "lluvia", "imerg", "gpm"],
    "evi": ["evi", "ndvi", "vegetation", "vegetacion", "vegetación", "modis evi"],
    "lst_day_c": ["lst", "land surface temperature", "surface temperature", "temperatura de superficie",
                  "temperatura superficial", "surface temp"],
    "t2m_c": ["t2m", "air temperature", "temperature", "temperatura", "temperatura del aire", "air temp"],
    "rh2m_pct": ["rh", "rh2m", "humidity", "relative humidity", "humedad", "humedad relativa"],
}


def _var(variable: str) -> str | None:
    v = variable.strip().lower().replace("-", " ").replace("_", " ")
    for key, names in ALIASES.items():
        if key not in GLOBAL_VARIABLES:
            continue
        if v == key.replace("_", " ") or v in names:
            return key
    return None


def _unknown(variable: str) -> dict:
    # el modelo lee esto y corrige el nombre en la siguiente llamada
    return {"ok": False, "reason": f"variable desconocida '{variable}'", "usa_una_de": list(GLOBAL_VARIABLES)}


def safe(fn):
    # un error en una herramienta vuelve al modelo como dato, no tumba la respuesta
    @wraps(fn)
    def run(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    return run


@tool
def listar_variables() -> dict:
    """Variables NASA disponibles: clave, nombre, unidades y si es solo tierra."""
    return {k: {"label": v["label"], "units": v["units"], "land_only": v["land_only"]}
            for k, v in GLOBAL_VARIABLES.items()}


@tool
@safe
def analizar_punto(variable: str, lat: float, lng: float) -> dict:
    """Tendencia 2001-2024 de una variable en un punto (Mann-Kendall estacional y pendiente de Sen)."""
    key = _var(variable)
    if not key:
        return _unknown(variable)
    return _short(clean(analyze_area(key, point={"lat": lat, "lng": lng})))


@tool
@safe
def analizar_area(variable: str, poligono: list[list[float]]) -> dict:
    """Tendencia 2001-2024 promediada en un polígono. poligono = [[lng, lat], ...] cerrado."""
    key = _var(variable)
    if not key:
        return _unknown(variable)
    try:
        return _short(clean(analyze_area(key, geometry={"type": "Polygon", "coordinates": [poligono]})))
    except ValueError as exc:
        return {"ok": False, "reason": str(exc)}


@lru_cache(maxsize=1)
def _forecast():
    if not FORECAST.exists():
        return None
    f = np.load(FORECAST)
    spread = f["spread"] if "spread" in f.files else None
    return f["data"], [str(m) for m in f["months"]], [str(v) for v in f["vars"]], spread


@tool
@safe
def prediccion(variable: str, lat: float, lng: float) -> dict:
    """Predicción mensual de la red neuronal EarthScope (después de 2024) en un punto, con promedios anuales."""
    key = _var(variable)
    if not key:
        return _unknown(variable)
    variable = key
    fc = _forecast()
    if fc is None:
        return {"ok": False, "reason": "la predicción de la red neuronal todavía no está cargada en el servidor"}
    data, months, vars_, spread = fc
    if variable not in vars_:
        return {"ok": False, "reason": f"la red no predice {variable}"}
    h, w = data.shape[-2:]
    y = min(int((90 - lat) / 180 * h), h - 1)
    x = min(int((lng + 180) / 360 * w), w - 1)
    serie = data[:, vars_.index(variable), y, x]
    if not np.isfinite(serie).any():
        return {"ok": False, "reason": "sin predicción en ese punto (por ejemplo, variable de tierra sobre el océano)"}
    anual: dict[str, list[float]] = {}
    for m, v in zip(months, serie):
        anual.setdefault(m[:4], []).append(float(v))
    res = {"ok": True, "units": GLOBAL_VARIABLES[variable]["units"], "model": "EarthScope U-Net, rollout mensual",
           "status": "experimental",
           "nota": "Validada solo a 1 mes vista; los años siguientes no están validados ni incluyen escenarios.",
           "annual_means": {k: round(float(np.mean(v)), 3) for k, v in anual.items()},
           "monthly": {m: round(float(v), 3) for m, v in zip(months, serie)}}
    if spread is not None:
        # dispersión del ensemble: cuanto más alta, menos confiable
        sp = spread[:, vars_.index(variable), y, x]
        res["uncertainty_annual"] = {k: round(float(np.mean(sp[[m[:4] == k for m in months]])), 3) for k in anual}
    return res


# acciones que la web ejecuta sobre el globo, una lista por petición
ACTIONS: ContextVar[list | None] = ContextVar("earthscope_actions", default=None)
MODES = {"valor": "value", "value": "value", "media": "value", "anomalia": "anomaly", "anomalía": "anomaly",
         "anomaly": "anomaly", "tendencia": "trend", "trend": "trend"}
ZOOM = {"cerca": 0.9, "medio": 1.5, "lejos": 2.4}


def _act(action: dict) -> None:
    acts = ACTIONS.get()
    if acts is not None:
        acts.append(action)


@tool
@safe
def mover_globo(lat: float, lng: float, zoom: str = "medio", seleccionar: bool = True) -> dict:
    """Gira el globo de la web hasta un lugar. zoom: cerca, medio o lejos.
    seleccionar=True marca el punto y abre su análisis en la web (úsalo cuando el usuario quiera ver un lugar)."""
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return {"ok": False, "reason": "coordenadas fuera de rango"}
    _act({"type": "fly", "lat": lat, "lng": lng, "altitude": ZOOM.get(zoom, 1.5)})
    if seleccionar:
        _act({"type": "select", "lat": lat, "lng": lng})
    return {"ok": True, "hecho": f"globo en {lat:.2f}, {lng:.2f}"}


@tool
@safe
def mostrar_capa(variable: str, modo: str = "valor", anio: int | None = None, animar_desde: int | None = None) -> dict:
    """Muestra una capa NASA en el globo de la web.
    modo: valor (media anual), anomalia (diferencia con lo normal) o tendencia (cambio por década 2001-2024).
    anio: año a mostrar: 2001-2024 son datos NASA; 2025-2030 es la predicción de la red neuronal (solo modo valor).
    animar_desde: año desde el que reproducir la animación año a año."""
    key = _var(variable)
    if not key:
        return _unknown(variable)
    mode = MODES.get(modo.strip().lower(), "value")
    act = {"type": "layer", "variable": key, "mode": mode}
    if animar_desde is not None:
        act["year"], act["play"] = max(2001, min(2024, int(animar_desde))), True
    elif anio is not None:
        act["year"] = max(2001, min(2030, int(anio)))
        if act["year"] > 2024:
            act["mode"] = "value"
    _act(act)
    return {"ok": True, "hecho": f"capa {key} en modo {mode}" + (" animada" if act.get("play") else "")}


@tool
@safe
def eventos_naturales(tipo: str = "", mostrar: bool = True) -> dict:
    """Eventos naturales activos ahora mismo según NASA EONET: incendios, tormentas, volcanes, inundaciones, hielo.
    tipo: filtra por fire, storm, volcano, flood, ice, dust, heat... (vacío = todos).
    mostrar: enciende la capa de eventos en el globo. Para llevar al usuario a uno, usa mover_globo con sus coordenadas."""
    from api import events as ev

    if mostrar:
        _act({"type": "events", "on": True})
    try:
        items = ev.get()["events"]
    except Exception:  # noqa: BLE001
        return {"ok": False, "reason": "no hay conexión con NASA EONET en este momento"}
    t = tipo.strip().lower()
    alias = {"incendio": "fire", "fuego": "fire", "tormenta": "storm", "huracán": "storm", "huracan": "storm",
             "ciclón": "storm", "volcán": "volcano", "volcan": "volcano", "inundación": "flood", "hielo": "ice"}
    t = next((v for k, v in alias.items() if t.startswith(k)), t)
    if t:
        items = [e for e in items if e["kind"] == t]
    counts: dict[str, int] = {}
    for e in items:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    return {"ok": True, "fuente": "NASA EONET", "total": len(items), "por_tipo": counts,
            "recientes": [{k: e[k] for k in ("title", "kind", "lat", "lng", "date") if k in e} for e in items[:12]]}


@tool
@safe
def zoom_globo(direccion: str = "acercar", veces: int = 1) -> dict:
    """Acerca o aleja la cámara del globo. direccion: acercar o alejar. veces: 1 a 3."""
    d = "in" if direccion.strip().lower().startswith(("acerc", "in", "zoom in", "cerca")) else "out"
    for _ in range(max(1, min(3, int(veces)))):
        _act({"type": "zoom", "dir": d})
    return {"ok": True, "hecho": f"zoom {d}"}


@tool
@safe
def controlar_vista(rotacion: str | None = None, imagen: str | None = None, animacion: str | None = None,
                    caso_amazonia: bool = False, ocultar_capa: bool = False, limpiar: bool = False,
                    tutorial: bool = False) -> dict:
    """Controla la vista del globo.
    rotacion: girar o parar (rotación automática del planeta).
    imagen: dia o noche (fondo del globo: satélite de día o luces nocturnas).
    animacion: reproducir o pausar (la línea de tiempo de la capa).
    caso_amazonia: True para ir al caso de estudio de la Amazonía central con mapas de alta resolución.
    ocultar_capa: True para quitar la capa de datos del globo.
    limpiar: True para quitar el punto o área seleccionada.
    tutorial: True para iniciar el recorrido guiado por la web."""
    hecho = []
    if rotacion:
        on = rotacion.strip().lower().startswith(("gir", "rot", "on", "si", "sí", "activ"))
        _act({"type": "spin", "on": on}); hecho.append("rotación " + ("activada" if on else "parada"))
    if imagen:
        night = imagen.strip().lower().startswith(("noc", "night"))
        _act({"type": "imagery", "mode": "night" if night else "day"}); hecho.append("noche" if night else "día")
    if animacion:
        on = animacion.strip().lower().startswith(("rep", "play", "anim", "on", "inic"))
        _act({"type": "play", "on": on}); hecho.append("animación " + ("en marcha" if on else "en pausa"))
    if caso_amazonia:
        _act({"type": "scope", "scope": "amazon"}); hecho.append("caso Amazonía")
    if ocultar_capa:
        _act({"type": "hide"}); hecho.append("capa oculta")
    if limpiar:
        _act({"type": "clear"}); hecho.append("selección limpia")
    if tutorial:
        _act({"type": "tour"}); hecho.append("tutorial iniciado")
    return {"ok": True, "hecho": ", ".join(hecho) or "nada"}


@lru_cache(maxsize=1)
def get_agent():
    # pocos reintentos y timeout corto: si el proveedor falla, el error llega a la web
    def load(m):
        return init_chat_model(m, max_retries=2, timeout=40) if isinstance(m, str) else m

    middleware = [ModelFallbackMiddleware(*[load(m) for m in FALLBACKS])] if FALLBACKS else []
    return create_agent(
        model=load(MODEL),
        middleware=middleware,
        tools=[listar_variables, analizar_punto, analizar_area, prediccion, mover_globo, mostrar_capa,
               zoom_globo, controlar_vista, eventos_naturales],
        system_prompt=PROMPT,
        checkpointer=InMemorySaver(),
    )


# variable de entorno que necesita cada proveedor
KEYS = {"anthropic": ["ANTHROPIC_API_KEY"], "google_genai": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY"], "groq": ["GROQ_API_KEY"], "openai": ["OPENAI_API_KEY"]}


def available() -> bool:
    if not isinstance(MODEL, str):
        return True
    need = KEYS.get(MODEL.split(":", 1)[0])
    return need is None or any(os.environ.get(k) for k in need)


# --- idioma: el modelo a veces contesta en el idioma del prompt (español) aunque le pregunten en inglés.
# Se detecta el idioma del mensaje y se le ordena explícitamente responder en ese idioma.
_ES = {"que", "qué", "el", "la", "los", "las", "de", "del", "en", "y", "es", "por", "para", "como", "cómo",
       "cuánto", "cuanto", "dónde", "donde", "muéstrame", "muestrame", "hola", "aquí", "aqui", "está", "esta",
       "lluvia", "vegetación", "temperatura", "hace", "más", "mas", "qué", "pasa", "gracias", "un", "una"}
_EN = {"what", "the", "is", "are", "of", "in", "and", "how", "where", "show", "me", "here", "this", "does",
       "rain", "vegetation", "temperature", "warmer", "hello", "hi", "thanks", "much", "a", "an", "to", "it"}


def detect_lang(message: str, fallback: str | None = None) -> str:
    """'es' o 'en' según las palabras del mensaje; si empata, el idioma de la interfaz (o inglés)."""
    words = re.findall(r"[a-záéíóúñü]+", message.lower())
    es = sum(w in _ES for w in words) + 2 * bool(re.search(r"[¿¡ñáéíóú]", message.lower()))
    en = sum(w in _EN for w in words)
    if es > en:
        return "es"
    if en > es:
        return "en"
    return fallback if fallback in ("es", "en") else "en"


def ask(message: str, session: str, context: dict | None = None) -> dict:
    lang = detect_lang(message, (context or {}).get("lang"))
    rule = ("[Responde SOLO en español.]" if lang == "es" else "[Reply ONLY in English, even though your instructions are in Spanish.]")
    text = f"{rule}\n{message}"
    if context:
        # lo que el usuario ve en el globo
        text = f"[contexto de la vista: {context}]\n{text}"
    actions: list = []
    token = ACTIONS.set(actions)
    try:
        out = get_agent().invoke({"messages": [{"role": "user", "content": text}]},
                                 {"configurable": {"thread_id": session}})
    finally:
        ACTIONS.reset(token)
    # las herramientas pueden correr en paralelo: orden fijo para la web
    order = {"clear": 0, "scope": 1, "fly": 2, "select": 3, "zoom": 4, "layer": 5, "hide": 6,
             "imagery": 7, "spin": 8, "play": 9, "events": 10, "tour": 11}
    actions.sort(key=lambda a: order.get(a["type"], 9))
    return {"reply": plain(out["messages"][-1].text), "actions": actions}


def plain(text: str) -> str:
    # quita el markdown que se cuele: negritas, cursivas, títulos y código
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.M)
    text = re.sub(r"^(\s*)\*\s+", r"\1- ", text, flags=re.M)
    return text.replace("`", "").strip()


# --- voz: el audio se transcribe con Gemini y la pregunta va al agente
# por defecto usa el mismo modelo de Gemini del agente (o su primer respaldo de Gemini)
def _stt_default() -> str:
    for m in [MODEL if isinstance(MODEL, str) else "", *FALLBACKS]:
        if m.startswith("google_genai:"):
            return m.split(":", 1)[1]
    return "gemini-2.5-flash-lite"


STT_MODEL = os.environ.get("EARTHSCOPE_STT_MODEL") or _stt_default()


def stt_available() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _stt_models() -> list[str]:
    # el modelo de voz y, si falla, los modelos de Gemini del agente
    out = [STT_MODEL]
    for m in [MODEL if isinstance(MODEL, str) else "", *FALLBACKS]:
        if m.startswith("google_genai:") and m.split(":", 1)[1] not in out:
            out.append(m.split(":", 1)[1])
    if "gemini-2.5-flash" not in out:
        out.append("gemini-2.5-flash")
    return out


# ElevenLabs Scribe v2: transcripción multilingüe (90+ idiomas, detecta el idioma solo).
# Se usa cuando hay ELEVENLABS_API_KEY; si no, o si falla, se transcribe con Gemini.
SCRIBE_MODEL = os.environ.get("EARTHSCOPE_SCRIBE_MODEL", "scribe_v2")
# palabras del proyecto que el modelo debe reconocer bien
KEYTERMS = ["EarthScope", "NASA", "MODIS", "MERRA-2", "GPM", "IMERG", "EVI", "Amazonía", "Cartagena", "Landsat"]


def _scribe(audio: bytes, mime: str) -> str:
    import httpx

    r = httpx.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        data={"model_id": SCRIBE_MODEL, "tag_audio_events": "false", "keyterms": KEYTERMS},
        files={"file": ("voz.wav", audio, mime)},
        timeout=30,
    )
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


def transcribe(audio: bytes, mime: str = "audio/wav") -> str:
    errors = []
    if os.environ.get("ELEVENLABS_API_KEY"):
        try:
            return _scribe(audio, mime)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{SCRIBE_MODEL}: {str(exc)[:160]}")
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError("; ".join(errors) or "no speech-to-text key")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    for model in _stt_models():
        try:
            r = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=audio, mime_type=mime),
                    "Transcribe exactamente lo que dice la persona, en su idioma. Devuelve solo el texto, sin comillas. "
                    "Si no se entiende nada o solo hay ruido, devuelve una cadena vacía.",
                ],
            )
            return (r.text or "").strip().strip('"')
        except Exception as exc:  # noqa: BLE001
            # modelo inexistente o saturado: se prueba el siguiente
            errors.append(f"{model}: {str(exc)[:120]}")
    raise RuntimeError("; ".join(errors))
