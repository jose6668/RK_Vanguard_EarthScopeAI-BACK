"""Agente: endpoint y herramientas, con un modelo falso (sin API key ni red)."""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("EARTHSCOPE_DATA", "sample")
os.environ.pop("ANTHROPIC_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from api import agent  # noqa: E402
from api.main import app  # noqa: E402

c = TestClient(app)
real_transcribe = agent.transcribe
body = {"message": "hola", "session": "t1"}

# sin key: 503
r = c.post("/api/v1/agent/chat", json=body)
assert r.status_code == 503, r.text

# herramientas
v = agent.listar_variables.invoke({})
assert "t2m_c" in v
p = agent.analizar_punto.invoke({"variable": "t2m_c", "lat": 10.4, "lng": -75.5})
assert p["ok"] and "monthly" not in p and p["units"] == "°C", p
# nombres alternativos y variable desconocida, sin excepción
assert agent.analizar_punto.invoke({"variable": "EVI", "lat": -5, "lng": -64})["ok"]
assert agent.analizar_punto.invoke({"variable": "Temperatura del aire", "lat": 10.4, "lng": -75.5})["ok"]
bad = agent.analizar_punto.invoke({"variable": "nada", "lat": 0, "lng": 0})
assert not bad["ok"] and "evi" in bad["usa_una_de"], bad

# predicción con un archivo de prueba
tmp = Path("/tmp/forecast_test.npz")
np.savez(tmp, data=np.full((24, 1, 90, 180), 20.0, np.float32),
         months=np.array([f"{2025 + i // 12}-{i % 12 + 1:02d}" for i in range(24)]), vars=np.array(["t2m_c"]))
agent.FORECAST = tmp
agent._forecast.cache_clear()
f = agent.prediccion.invoke({"variable": "t2m_c", "lat": 10, "lng": -75})
assert f["ok"] and f["annual_means"] == {"2025": 20.0, "2026": 20.0}, f


# modelo falso: pide una herramienta y luego responde
class Fake(BaseChatModel):
    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if isinstance(messages[-1], ToolMessage):
            msg = AIMessage(content=f"listo: {messages[-1].content[:40]}")
        else:
            msg = AIMessage(content="", tool_calls=[{"name": "listar_variables", "args": {}, "id": "c1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


agent.MODEL = Fake()
agent.get_agent.cache_clear()
os.environ["ANTHROPIC_API_KEY"] = "test"
r = c.post("/api/v1/agent/chat", json={**body, "context": {"layer": "t2m_c", "point": [10.4, -75.5]}})
assert r.status_code == 200, r.text
assert r.json()["reply"].startswith("listo:"), r.json()

# respaldo: si el modelo principal falla, responde el siguiente
class Down(Fake):
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        raise RuntimeError("503 saturado")


agent.MODEL, agent.FALLBACKS = Down(), [Fake()]
agent.get_agent.cache_clear()
r = c.post("/api/v1/agent/chat", json={**body, "session": "t2"})
assert r.status_code == 200 and r.json()["reply"].startswith("listo:"), r.text

# acciones sobre el globo
class Mover(Fake):
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if isinstance(messages[-1], ToolMessage):
            msg = AIMessage(content="**Mira** la Amazonía")
        else:
            msg = AIMessage(content="", tool_calls=[
                {"name": "mover_globo", "args": {"lat": -3.5, "lng": -62, "zoom": "cerca"}, "id": "m1"},
                {"name": "mostrar_capa", "args": {"variable": "vegetación", "modo": "tendencia", "animar_desde": 2005},
                 "id": "m2"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


agent.MODEL, agent.FALLBACKS = Mover(), []
agent.get_agent.cache_clear()
r = c.post("/api/v1/agent/chat", json={**body, "session": "t3"}).json()
assert r["reply"] == "Mira la Amazonía", r
types = [a["type"] for a in r["actions"]]
assert types == ["fly", "select", "layer"], r["actions"]
lay = r["actions"][2]
assert lay["variable"] == "evi" and lay["mode"] == "trend" and lay["year"] == 2005 and lay["play"], lay
# control de la vista
assert agent.zoom_globo.invoke({"direccion": "acercar", "veces": 2})["ok"]
v = agent.controlar_vista.invoke({"rotacion": "parar", "imagen": "noche", "animacion": "reproducir"})
assert v["ok"] and "noche" in v["hecho"], v
# eventos naturales (EONET simulado) y año futuro en la capa
import api.events as ev  # noqa: E402
ev._mem.update(t=9e18, data={"events": [{"id": "1", "title": "Fire A", "kind": "fire", "lat": 1, "lng": 2, "date": "2026-11-10"},
                                         {"id": "2", "title": "Storm B", "kind": "storm", "lat": 3, "lng": 4, "date": "2026-11-11"}]})
e = agent.eventos_naturales.invoke({"tipo": "incendios"})
assert e["ok"] and e["total"] == 1 and e["recientes"][0]["title"] == "Fire A", e
assert agent.mostrar_capa.invoke({"variable": "vegetación", "modo": "tendencia", "anio": 2029})["ok"]
ev._mem.clear()
# fuera de una petición, las herramientas no fallan
assert agent.mover_globo.invoke({"lat": 0, "lng": 0})["ok"]

# voz: sin key de Google responde 503; con transcripción simulada pasa por el agente
os.environ.pop("GOOGLE_API_KEY", None)
assert c.post("/api/v1/agent/voice", content=b"RIFF", headers={"content-type": "audio/wav"}).status_code == 503
os.environ["GOOGLE_API_KEY"] = "test"
agent.transcribe = lambda audio, mime="audio/wav": "llévame a la Amazonía"
r = c.post("/api/v1/agent/voice?session=v1", content=b"RIFF....", headers={"content-type": "audio/wav"})
assert r.status_code == 200 and r.json()["transcript"] == "llévame a la Amazonía" and r.json()["actions"], r.text
agent.transcribe = lambda audio, mime="audio/wav": ""
assert c.post("/api/v1/agent/voice", content=b"RIFF", headers={"content-type": "audio/wav"}).json()["reply"] == ""

r = c.post("/api/v1/agent/chat", json={"message": "", "session": "t1"})
assert r.status_code == 422

# transcripción: si el primer modelo no existe, prueba el siguiente
import google.genai as gg  # noqa: E402

calls = []


class _Models:
    def generate_content(self, model, contents):
        calls.append(model)
        if model == "no-existe":
            raise RuntimeError("404 NOT_FOUND")
        return type("R", (), {"text": "hola mundo"})()


class _Client:
    def __init__(self, **kw):
        self.models = _Models()



gg.Client = _Client
agent.STT_MODEL = "no-existe"
agent.MODEL, agent.FALLBACKS = "google_genai:modelo-bueno", []
os.environ["GOOGLE_API_KEY"] = "x"
assert real_transcribe(b"RIFF") == "hola mundo" and calls == ["no-existe", "modelo-bueno"], calls

# con ELEVENLABS_API_KEY se usa Scribe; si falla, Gemini
import httpx  # noqa: E402

sent = {}


def _post(url, headers, data, files, timeout):
    sent.update(url=url, model=data["model_id"], keyterms=data["keyterms"])
    return httpx.Response(200, json={"text": "muéstrame la Amazonía"}, request=httpx.Request("POST", url))


httpx.post = _post
os.environ["ELEVENLABS_API_KEY"] = "x"
assert real_transcribe(b"RIFF") == "muéstrame la Amazonía", sent
assert sent["model"] == "scribe_v2" and "EarthScope" in sent["keyterms"]


def _fail(*a, **k):
    raise httpx.ConnectError("sin red")


httpx.post = _fail
calls.clear()
assert real_transcribe(b"RIFF") == "hola mundo" and calls == ["no-existe", "modelo-bueno"], calls
del os.environ["ELEVENLABS_API_KEY"]

print("test_agent: OK")
