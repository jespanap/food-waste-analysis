# FIKWaste — Waste Generation Dataset from Three Restaurant Kitchens in Portugal

Datos reales de generación de residuos en **3 cocinas de restaurante** en Portugal, medidos con
sensores ultrasónicos instalados en las tapas de los contenedores.

- **Fuente:** https://osf.io/tyaj6/ (OSF, público)
- **Paper:** Pereira, L.; Aguiar, V.; Vasconcelos, F. *FIKWaste: A Waste Generation Dataset from
  Three Restaurant Kitchens in Portugal.* Data 2021, 6, 25. https://doi.org/10.3390/data6030025
  (incluido en `FIKWaste_paper_Data2021.pdf`)
- **Licencia:** CC BY 4.0 — uso libre con atribución.
- **Descargado:** 2026-08-13

---

## Estructura

```
FIKWaste/
├── deployments.csv              # metadatos de las 3 cocinas
├── FIKWaste_paper_Data2021.pdf  # paper con metodología completa
├── Kitchen 1/
│   ├── Paper/{measurements.csv, labels.csv}
│   ├── Plastic/{measurements.csv, labels.csv}
│   └── Undifferentiated/{measurements.csv, labels.csv}
├── Kitchen 2/   (Glass, Paper, Plastic, Undifferentiated)
└── Kitchen 3/   (Glass, Paper, Plastic, Undifferentiated)
```

**Ojo: Kitchen 1 no tiene contenedor de vidrio monitoreado.** Son 11 combinaciones
cocina×contenedor, no 12.

---

## Esquema

### `measurements.csv` — serie temporal cruda por contenedor

| Columna     | Descripción                                        | Unidad   |
|-------------|----------------------------------------------------|----------|
| `timestamp` | Momento en que se activó el sensor                 | datetime |
| `distance`  | Distancia entre el sensor y la superficie del residuo | cm    |
| `volume`    | Volumen de residuo correspondiente                 | %        |

`volume = (A_base × H_sensor) / (A_base × H_bin) × 100`

### `labels.csv` — eventos de vaciado del contenedor (ground truth)

| Columna     | Descripción                                   | Unidad   |
|-------------|-----------------------------------------------|----------|
| `timestamp` | Timestamp correspondiente en `measurements.csv` | datetime |
| `volume`    | Volumen en ese momento                        | %        |
| `source`    | Origen de la etiqueta: `V` = vídeo, `H` = humano | texto |

Los archivos de labels traen **BOM UTF-8** → leer con `encoding='utf-8-sig'`.

### `deployments.csv` — metadatos por cocina

`ID`, `service` (Dinner / Breakfast and Dinner), `area` (m²), `capacity` (comensales
simultáneos), `has_glass`/`glass_volume` (m³) y equivalentes para paper/plastic/
undifferentiated, `start`, `end` (formato `DD/MM/YYYY`).

---

## Contenido real (verificado tras la descarga)

| Cocina    | Contenedor       | Mediciones | Labels | Inicio              | Fin                 |
|-----------|------------------|-----------:|-------:|---------------------|---------------------|
| Kitchen 1 | Paper            | 19.365     | 12     | 2019-02-06 09:59:43 | 2019-03-01 17:26:09 |
| Kitchen 1 | Plastic          | 29.246     | 12     | 2019-02-06 09:58:41 | 2019-03-03 21:22:23 |
| Kitchen 1 | Undifferentiated | 10.036     | 10     | 2019-02-06 00:00:38 | 2019-03-03 11:28:44 |
| Kitchen 2 | Glass            |  3.398     |  6     | 2019-03-12 12:15:33 | 2019-04-02 17:42:56 |
| Kitchen 2 | Paper            |  3.024     | 10     | 2019-03-12 12:16:48 | 2019-04-01 21:57:40 |
| Kitchen 2 | Plastic          |  3.403     |  8     | 2019-03-12 12:16:52 | 2019-04-02 01:57:51 |
| Kitchen 2 | Undifferentiated |  1.456     |  5     | 2019-03-12 12:14:43 | 2019-04-02 21:28:16 |
| Kitchen 3 | Glass            |  4.313     |  4     | 2019-04-16 19:13:28 | 2019-05-15 00:55:52 |
| Kitchen 3 | Paper            |  4.307     | 11     | 2019-04-16 17:33:57 | 2019-05-10 10:51:47 |
| Kitchen 3 | Plastic          |  5.433     | 14     | 2019-04-16 17:34:02 | 2019-05-15 00:55:13 |
| Kitchen 3 | Undifferentiated |  2.063     | 14     | 2019-04-16 17:34:43 | 2019-05-14 16:47:59 |
| **Total** |                  | **86.044** | **106**|                     |                     |

---

## Advertencias metodológicas (del paper, importantes para el análisis)

1. **Frecuencia de muestreo distinta por cocina.** Kitchen 1 muestrea cada **1 min**;
   Kitchens 2 y 3 cada **5 min** (para ahorrar batería). No compares conteos de muestras
   entre cocinas sin normalizar.
2. **Kitchens 2 y 3 solo capturan durante horario laboral.** Kitchen 1 captura 24 h.
3. **`volume` rara vez llega a 0 tras un vaciado.** Las bolsas vacías no se estiran del todo
   y el peso del residuo (sobre todo papel y plástico) impide que llegue al fondo. No uses
   `volume ≈ 0` para detectar vaciados — para eso están los `labels.csv`.
4. **Bajadas de volumen ≠ vaciado.** A veces el personal solo reacomodaba la bolsa.
5. **Los labels están incompletos.** Solo se conservaron eventos etiquetados por al menos 2
   de los 3 autores; el vídeo no estuvo disponible todo el tiempo. Hay vaciados reales sin
   etiquetar.
6. **Es volumen (%), no peso.** No hay kg. Si necesitas masa, tendrás que asumir densidades
   por fracción y `*_volume` (m³) de `deployments.csv`.
7. Es residuo total por fracción, no "food waste" puro — la fracción orgánica/alimentaria
   corresponde a `Undifferentiated`.

---

## Carga rápida en pandas

```python
import pandas as pd, glob, os, re

rows = []
for path in glob.glob("Kitchen */*/measurements.csv"):
    kitchen, bin_type = path.split(os.sep)[:2]
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["kitchen"], df["bin"] = kitchen, bin_type
    rows.append(df)
meas = pd.concat(rows, ignore_index=True)

labels = []
for path in glob.glob("Kitchen */*/labels.csv"):
    kitchen, bin_type = path.split(os.sep)[:2]
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["timestamp"])
    df["kitchen"], df["bin"] = kitchen, bin_type
    labels.append(df)
labels = pd.concat(labels, ignore_index=True)

deployments = pd.read_csv("deployments.csv", dayfirst=True,
                          parse_dates=["start", "end"])
```

---

## Datasets hermanos (mismas 3 cocinas)

Del mismo proyecto FIK, permiten cruzar residuos con otros consumos:

- **FIKElectricity** — consumo eléctrico: https://www.nature.com/articles/s41597-023-02698-8
- **FIKWater** — consumo de agua (caudalímetros ultrasónicos, agua fría y caliente, 1/5 Hz):
  paper https://doi.org/10.3390/data6030026 · datos https://osf.io/7bz2m/
