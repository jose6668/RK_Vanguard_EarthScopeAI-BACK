# Simulaciones experimentales ("¿y si…?")

Base para la siguiente fase de EarthScope AI. Un **escenario** responde *"si cambia X, ¿qué estimaría un modelo para Y?"*. No es una observación ni un pronóstico, y la API lo marca siempre con `data_kind: "scenario"` y `status: "experimental"`.

## Reglas (las aplica `science/simulation.py`)

1. **Puerta de validación.** Un modelo solo se puede simular si su R² fuera de muestra (leave-one-year-out) es ≥ `SKILL_THRESHOLD` (0,3). Si no llega, queda listado como `no_skill`, que también es un resultado, y `POST /simulate` responde 409.
2. **Incertidumbre siempre.** Cada respuesta trae el intervalo de predicción al 95 %, el rango observado de la entrada, `extrapolation: true/false`, los supuestos y una advertencia.
3. **Dominio.** Cada modelo pertenece a una región y una temporada. No se aplica fuera de ellas.
4. **La web no calcula ciencia.** El frontend solo llama a la API (`web/lib/simulate.ts`) y muestra lo que recibe.

## Modelos registrados

| id | Pregunta | Validación | Estado |
|---|---|---|---|
| `amazon_dry_season_lst` | Si la lluvia de la estación seca (jul–oct) de la Amazonía central cambiara un X %, ¿qué temperatura de superficie estimaría el modelo? | LOYO R² ≈ 0,49 (2001–2024) | `validated` |
| EVI vs lluvia | ¿La vegetación responde a la lluvia de la estación seca? | R² ≤ 0,1 | Sin habilidad: no se ofrece (resultado negativo documentado) |

Ejemplo: `-20 %` de lluvia da `+0,47 °C` sobre la línea base, con un intervalo de 95 % de unos ±0,6 °C. A `-50 %` se sale del rango observado y la respuesta marca `extrapolation: true`.

## API

```bash
curl localhost:8000/api/v1/simulate/models
curl -X POST localhost:8000/api/v1/simulate -H 'content-type: application/json' \
  -d '{"model":"amazon_dry_season_lst","inputs":{"precip_change_pct":-20}}'
```

## Cómo agregar un modelo (HU nueva)

1. Escribe la ciencia en `science/` (ajuste, validación fuera de muestra y `simulate`) con pruebas sintéticas.
2. Regístralo en `science/simulation.py` con `register(ScenarioModel(...))`. Define los `inputs` con unidades y rango, las fuentes con DOI y los supuestos.
3. Añade sus casos a `tests/test_simulation.py`.
4. Si no pasa la puerta de validación, regístralo igual: se mostrará como `no_skill`.

Ideas para la siguiente fase: escenarios de temperatura del aire (MERRA-2) por región, sensibilidad de humedad relativa y escenarios combinados (lluvia + temperatura). Cada uno debe pasar por la misma puerta.
