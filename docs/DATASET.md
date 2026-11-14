# Dataset global de EarthScope AI (en git)

**Al clonar este repo ya tienes el dataset completo.** No hace falta una cuenta de NASA ni descargar nada.

| | |
|---|---|
| Carpeta | `data/dataset/` |
| Contenido | 8 variables × 24 años (2001–2024) × 12 meses, todo el planeta |
| Grilla | 0,5° · 360 × 720 celdas · norte→sur, oeste→este · primera celda 89,75° N, 179,75° W |
| Tamaño | **369 MB comprimido** (≈ 1,9 GB sin comprimir) en 192 archivos de ~1–3 MB |
| Lector | `science/dataset.py` → `load_cube(var)`, `load_year(var, año)`, `months(var)`, `grid()` |
| Construcción | `pipeline/fetch_global.py` → `pipeline/pack_dataset.py` |

## Variables

| Clave | Variable | Unidad | Precisión guardada | Producto NASA | Tipo | MB |
|---|---|---|---|---|---|---|
| `precip_mm_day` | Precipitación | mm/día | 0,01 | GPM IMERG V07 (GPM_3IMERGM) | Satélite + pluviómetros | 61 |
| `ndvi` | Vegetación NDVI | índice | 0,0001 | MODIS MOD13C2 v6.1 | Satélite, solo tierra | 30 |
| `evi` | Vegetación EVI | índice | 0,0001 | MODIS MOD13C2 v6.1 | Satélite, solo tierra | 29 |
| `lst_day_c` | Temperatura de superficie, día | °C | 0,01 | MODIS MOD11C3 v6.1 | Satélite, solo tierra, cielo despejado | 35 |
| `lst_night_c` | Temperatura de superficie, noche | °C | 0,01 | MODIS MOD11C3 v6.1 | Satélite, solo tierra, cielo despejado | 35 |
| `t2m_c` | Temperatura del aire a 2 m | °C | 0,01 | MERRA-2 M2TMNXSLV v5.12.4 | **Reanálisis** | 51 |
| `qv2m_gkg` | Humedad específica a 2 m | g/kg | 0,001 | MERRA-2 M2TMNXSLV v5.12.4 | **Reanálisis** | 63 |
| `rh2m_pct` | Humedad relativa a 2 m | % | 0,01 | MERRA-2 (derivada con la fórmula de Magnus) | **Reanálisis** | 65 |

Los detalles de cada variable (DOI, años, error máximo, huella `sha`) están en `data/dataset/manifest.json`.

## Formato y compresión

```
data/dataset/<variable>/<AAAA>.i16.xz      12 meses de una variable
```
1. **Entero de 16 bits con escala**, como el `scale_factor` de los archivos de NASA: `valor ≈ entero × escala`. Los datos faltantes se guardan como −32768.
2. **Delta a lo largo de la longitud:** las celdas vecinas se parecen, así que se guarda la diferencia.
3. **Byte shuffle:** separa los bytes altos y bajos.
4. **LZMA (xz)**, el compresor más fuerte de la biblioteca estándar de Python.

| Método (un año de temperatura del aire) | Tamaño |
|---|---|
| Sin comprimir (int16) | 6,2 MB |
| zlib (`.npz` normal) | 4,1 MB |
| LZMA | 2,9 MB |
| **Delta + shuffle + LZMA (el usado)** | **2,1 MB** |

**Sin pérdida a la precisión guardada:** al empaquetar, cada archivo se decodifica de nuevo y se compara con el original. El error máximo es medio paso de la escala (por ejemplo 0,005 °C), mucho menor que la incertidumbre de los propios productos NASA. La máscara de datos faltantes se conserva exacta.

## Uso

```python
from science import dataset
cube, months = dataset.load_cube("t2m_c")      # (288, 360, 720) float32, NaN = sin dato
lat, lon = dataset.grid()
july_2024 = cube[months.index("2024-07")]
```

`load_cube` decodifica una sola vez y guarda un cubo mapeado en memoria en `data/.cache/` (fuera de git). `load_cube(var, use_cache=False)` lo devuelve en memoria sin escribir caché.

## Regenerarlo o ampliarlo

```bash
pip install -r requirements.txt                   # incluye earthaccess, h5py y pyhdf
python pipeline/fetch_global.py --product all     # ≈ 69 GB de descarga, ~1 h; deja data/global/ (fuera de git)
python pipeline/pack_dataset.py                   # → data/dataset/ (verifica cada archivo)
python pipeline/make_sample.py                    # → data/sample/ (2°, para CI)
python export/build_global_data.py                # → capas de la web (repo Frontend)
```

Requiere una cuenta [Earthdata Login](https://urs.earthdata.nasa.gov) con **"NASA GESDISC DATA ARCHIVE"** y **"LP DAAC Data Pool"** aprobadas, y `~/.netrc`.

Para agregar una variable: extráela en `pipeline/fetch_global.py`, regístrala en `science/dataset.py` (`PACKING`, con su escala) y, si se mostrará en la web, en `science/variables.py` (`GLOBAL_VARIABLES`).
