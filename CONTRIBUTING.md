# Cómo trabajar en el Backend

## 1. Ramas
- `main`: versión estable (la que usa el demo). Solo entra por PR desde `Develop`.
- `Develop`: integración del equipo. **Toda rama sale de `Develop` y vuelve a `Develop` por PR.**
- Nombre de rama: `<tipo>/HU-<n>-<descripcion>` (por ejemplo `feat/HU-07-valor-celda`). Tipos: `feat`, `fix`, `data`, `docs`, `chore`.

```bash
git checkout Develop && git pull
git checkout -b feat/HU-07-valor-celda
# … cambios + pruebas …
git push -u origin feat/HU-07-valor-celda      # y abre el PR hacia Develop
```

## 2. Reglas
| Regla | Por qué |
|---|---|
| Toda la ciencia va en `science/`. `api/` y `export/` solo la llaman. | Una sola fuente de verdad, con pruebas |
| **No romper el contrato de la API** (`tests/test_api.py`). Si cambias un campo, abre también un PR en el Frontend (`web/lib/api.ts`, `InsightPanel.tsx`) y únanlos juntos. | La web depende de ese formato |
| Si regeneras datos: `pipeline/pack_dataset.py` → `pipeline/make_sample.py` → exportadores, y sube `data/dataset/` y `data/sample/`. | El dataset en git debe estar siempre al día |
| No subir `data/global/` (crudo), `data/.cache/`, `.env` ni archivos de NASA sin procesar. | Pesan GB o son privados |
| Cada variable nueva va en `science/variables.py` (`GLOBAL_VARIABLES`) y en `science/dataset.py` (`PACKING`), con fuente, DOI y advertencias. | Trazabilidad (RN-12) |
| Las conclusiones usan el dataset completo, nunca la muestra. | La muestra de 2° es solo para desarrollo |

## 3. Antes del PR
```bash
export EARTHSCOPE_DATA=sample
python tests/test_trends.py && python tests/test_whatif.py && python tests/test_global_trends.py
python tests/test_area.py && python tests/test_api.py
EARTHSCOPE_DATA=full python tests/test_dataset.py
```

## 4. Protección de ramas (la configura el dueño del repo)
*Settings → Branches → Add branch ruleset* para `main` y `Develop`:
- ✅ Require a pull request before merging (1 aprobación)
- ✅ Require status checks to pass: `Backend (science + API)`
- ✅ Block force pushes

## 5. Historias de usuario
Formato, rúbrica (≥ 13/16), Definición de Listo y de Hecho: [RUBRICA_HISTORIAS_DE_USUARIO.md](https://github.com/junher30/RK_Vanguard_EarthScopeAI/blob/main/docs/gestion/RUBRICA_HISTORIAS_DE_USUARIO.md). Usa la plantilla de issue **"Historia de usuario"**.
