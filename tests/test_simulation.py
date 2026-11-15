"""Base de simulaciones: contrato y puerta de validación (python tests/test_simulation.py)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("EARTHSCOPE_DATA", "sample")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from science import simulation  # noqa: E402

c = TestClient(app)
models = c.get("/api/v1/simulate/models").json()["models"]
amazon = next(m for m in models if m["id"] == "amazon_dry_season_lst")
assert amazon["status"] == "validated" and amazon["validation"]["value"] >= simulation.SKILL_THRESHOLD, amazon
assert amazon["data_kind"] == "scenario"

r = c.post("/api/v1/simulate", json={"model": "amazon_dry_season_lst", "inputs": {"precip_change_pct": -20}}).json()
assert r["data_kind"] == "scenario" and r["status"] == "experimental"
lo, hi = r["prediction_interval95"]
assert lo < r["estimate"] < hi and r["delta_vs_baseline"] > 0      # menos lluvia → superficie más caliente
assert r["extrapolation"] is False
assert c.post("/api/v1/simulate", json={"model": "amazon_dry_season_lst", "inputs": {"precip_change_pct": -50}}).json()["extrapolation"] is True
assert c.post("/api/v1/simulate", json={"model": "amazon_dry_season_lst", "inputs": {"precip_change_pct": -99}}).status_code == 422
assert c.post("/api/v1/simulate", json={"model": "nope"}).status_code == 404

# puerta de validación: un modelo sin habilidad no se puede simular
simulation.register(simulation.ScenarioModel(
    id="_no_skill", label="t", question="t", region="t", output="t", output_units="t",
    inputs=[simulation.ScenarioInput("x", "x", "%", -1, 1)], method="t", sources=[], assumptions=[],
    fit=lambda: {"loyo_r2": 0.05}, run=lambda m, i: {}))
assert c.post("/api/v1/simulate", json={"model": "_no_skill"}).status_code == 409
print(f"OK - simulation registry: {amazon['label']} (R² {amazon['validation']['value']:.2f}); "
      f"-20 % rain → {r['delta_vs_baseline']:+.2f} °C [{lo:.2f}, {hi:.2f}]; no-skill models are blocked")
