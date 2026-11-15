"""Base para SIMULACIONES EXPERIMENTALES ("¿y si…?") de EarthScope AI.

Un escenario responde "si cambia X, ¿qué estimaría un modelo para Y?". No es una observación ni
un pronóstico: es un `scenario` (data_kind) y así se etiqueta siempre.

Reglas que el registro hace cumplir (las mismas del proyecto):
  1. Un modelo solo se puede SIMULAR si pasó validación fuera de muestra (`skill_threshold`).
     Los que no la pasan quedan registrados como `no_skill` (resultado negativo, también se muestra).
  2. Toda respuesta trae intervalo de predicción, rango observado, alerta de extrapolación,
     supuestos y advertencias.
  3. Los modelos son por región/dominio: no se aplican fuera de él.

Para agregar un modelo nuevo:
  - escribe la ciencia en science/ (ajuste + validación + simulate) con pruebas sintéticas;
  - regístralo abajo con un `ScenarioModel` (inputs con unidades y rango permitido);
  - `tests/test_simulation.py` comprueba el contrato y la puerta de validación.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

from science import whatif
from science.trends import load_series

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SKILL_THRESHOLD = 0.3          # R² fuera de muestra mínimo para permitir simular


@dataclass(frozen=True)
class ScenarioInput:
    key: str
    label: str
    units: str
    min: float                 # rango que acepta la API (más allá del observado se marca extrapolación)
    max: float
    default: float = 0.0


@dataclass
class ScenarioModel:
    id: str
    label: str
    question: str
    region: str
    output: str                # variable estimada
    output_units: str
    inputs: list[ScenarioInput]
    method: str
    sources: list[str]
    assumptions: list[str]
    fit: Callable[[], dict]                      # devuelve el modelo ajustado (con métricas de validación)
    run: Callable[[dict, dict], dict]            # (modelo, inputs) -> resultado
    skill_key: str = "loyo_r2"
    notes: list[str] = field(default_factory=list)

    @property
    def fitted(self) -> dict:
        return _fitted(self.id)

    def status(self) -> str:
        return "validated" if self.fitted.get(self.skill_key, -1) >= SKILL_THRESHOLD else "no_skill"

    def describe(self) -> dict:
        f = self.fitted
        return {
            "id": self.id, "label": self.label, "question": self.question, "region": self.region,
            "output": self.output, "output_units": self.output_units, "method": self.method,
            "inputs": [i.__dict__ for i in self.inputs], "sources": self.sources,
            "assumptions": self.assumptions, "notes": self.notes,
            "status": self.status(), "skill_threshold": SKILL_THRESHOLD,
            "validation": {"metric": self.skill_key, "value": f.get(self.skill_key),
                           "n": f.get("n_years"), "period": f.get("years")},
            "data_kind": "scenario",
        }


# ------------------------------------------------------------------ modelos registrados
def _fit_amazon_lst() -> dict:
    p = load_series(str(PROCESSED / "amazon_core_precipitation.csv"), "precip_mm_day")
    t = load_series(str(PROCESSED / "amazon_core_lst.csv"), "lst_day_c")
    return whatif.fit(whatif.seasonal_table(p, t))


def _run_amazon_lst(model: dict, inputs: dict) -> dict:
    r = whatif.simulate(model, float(inputs["precip_change_pct"]))
    return {
        "estimate": r["lst_estimate"], "delta_vs_baseline": r["delta_vs_baseline"],
        "baseline": model["baseline_lst"], "mean_ci95": r["mean_ci95"],
        "prediction_interval95": r["prediction_interval95"], "extrapolation": r["extrapolation"],
        "observed_input_range": r["observed_range_pct"], "input_effective": {"precip_mm_day": r["precip_mm_day"]},
    }


MODELS: dict[str, ScenarioModel] = {}


def register(m: ScenarioModel) -> None:
    MODELS[m.id] = m


register(ScenarioModel(
    id="amazon_dry_season_lst",
    label="Dry-season surface temperature vs rainfall (central Amazon)",
    question="If dry-season (Jul–Oct) rainfall changed by X %, what land-surface temperature would the model estimate?",
    region="amazon_core",
    output="lst_day_c", output_units="°C",
    inputs=[ScenarioInput("precip_change_pct", "Dry-season rainfall change", "%", -60, 60, -20)],
    method="OLS LST = a + b·r on 2001–2024 dry seasons; leave-one-year-out validation; 95 % prediction interval",
    sources=["GPM IMERG V07 (10.5067/GPM/IMERG/3B-MONTH/07)", "MODIS MOD11C3 v6.1 (10.5067/MODIS/MOD11C3.061)"],
    assumptions=whatif.ASSUMPTIONS,
    fit=_fit_amazon_lst, run=_run_amazon_lst,
    notes=["Vegetation (EVI) vs rainfall was tested and has NO out-of-sample skill (R² ≤ 0.1): not offered."],
))


@lru_cache(maxsize=None)
def _fitted(model_id: str) -> dict:
    return MODELS[model_id].fit()


# ------------------------------------------------------------------ API pública
def list_models() -> list[dict]:
    return [m.describe() for m in MODELS.values()]


def simulate(model_id: str, inputs: dict) -> dict:
    if model_id not in MODELS:
        raise KeyError(model_id)
    m = MODELS[model_id]
    if m.status() != "validated":
        raise PermissionError(f"model '{model_id}' did not pass out-of-sample validation; it cannot be simulated")
    clean_inputs = {}
    for i in m.inputs:
        v = float(inputs.get(i.key, i.default))
        if not (i.min <= v <= i.max):
            raise ValueError(f"{i.key} must be between {i.min} and {i.max} {i.units}")
        clean_inputs[i.key] = v
    res = m.run(m.fitted, clean_inputs)
    return {
        "model": m.id, "label": m.label, "region": m.region, "inputs": clean_inputs,
        "output": m.output, "units": m.output_units, **res,
        "status": "experimental", "data_kind": "scenario",
        "caveat": "Statistical scenario from an observed association: not an observation, not a forecast, not causal proof.",
        "assumptions": m.assumptions,
        "validation": m.describe()["validation"],
    }
