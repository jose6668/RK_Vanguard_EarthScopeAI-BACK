## Qué cambia
<!-- Una o dos frases. -->

## Historia de usuario
<!-- HU-XX · Closes #issue · si es "Ambos", enlaza el PR del Frontend -->

## Cómo lo probaste
- [ ] `EARTHSCOPE_DATA=sample` → `python tests/test_trends.py`, `test_whatif.py`, `test_global_trends.py`, `test_area.py`, `test_api.py`
- [ ] `EARTHSCOPE_DATA=full python tests/test_dataset.py`
- [ ] Probé la API en http://localhost:8000/docs

## Checklist
- [ ] La ciencia está en `science/` (la API y el frontend solo transportan y muestran)
- [ ] Si cambié la respuesta de la API: hay PR en el Frontend (`web/lib/api.ts`, `InsightPanel.tsx`) y `tests/test_api.py` lo cubre
- [ ] Si cambié `export/`: volví a exportar y subí `web/public/data/` en el Frontend
- [ ] Si regeneré datos: `python pipeline/pack_dataset.py` y `python pipeline/make_sample.py`
- [ ] No subí `data/global/`, `data/.cache/`, `.env` ni archivos crudos de NASA
