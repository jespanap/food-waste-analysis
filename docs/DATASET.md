# Descripción del dataset — FIKWaste

Documento requerido por la entrega **M1** (fuente, tamaño, idioma, licencia, tarea y sesgos).

---

## 1 · Fuente

| | |
|---|---|
| **Nombre** | FIKWaste — *A Waste Generation Dataset from Three Restaurant Kitchens in Portugal* |
| **Repositorio** | https://osf.io/tyaj6/ (Open Science Framework, acceso público) |
| **Paper** | Pereira, L.; Aguiar, V.; Vasconcelos, F. *Data* **2021**, 6, 25 · https://doi.org/10.3390/data6030025 |
| **Licencia** | **CC BY 4.0** — uso libre, incluido comercial, con atribución. Sin restricciones para este trabajo. |
| **Descarga** | 13 de agosto de 2026, desde el endpoint oficial de OSF (`files.osf.io/v1/resources/tyaj6/…`) |
| **Copia en el repo** | `data/raw/FIKWaste/` (25 archivos, 3,1 MB) + el paper en PDF |

Los datos **no son sintéticos ni simulados**: provienen de sensores ultrasónicos HC-SR04 instalados
en las tapas de los contenedores de basura de tres cocinas de restaurante reales en Portugal, durante
el proyecto *Future Industrial Kitchen* (FIK). Los sensores miden la distancia entre la tapa y la
superficie del residuo; esa distancia se convierte en un porcentaje de llenado.

### Por qué cambiamos de dataset (auditoría del anterior)

La versión anterior de este proyecto (ver `legacy/`) usaba `food_wastage_data.csv`, descargado de
Kaggle. Antes de descartarlo lo **auditamos**, y los resultados justifican el cambio:

| Prueba | Resultado | Interpretación |
|---|---|---|
| F1-macro alcanzable (HistGB, CV 5-fold) | **0,911 ± 0,009** | Absurdamente alto para predicción de desperdicio con 10 variables |
| Lo mismo con **etiquetas permutadas** | 0,326 | El modelo sí aprende algo real de las columnas, no es ruido |
| Combinaciones de features repetidas | 174 grupos (396 filas) | |
| …de esas, con target **idéntico** | **68,4 %** (dispersión mediana = 0) | El target es casi una **función determinista** de las features |
| Filas duplicadas exactas | 164 | |
| Valores faltantes | 0 | Ningún dato operativo real está tan limpio |
| Marca temporal | no existe | No se puede validar temporalmente |
| Fuente primaria citada | **ninguna** | |
| Licencia | **no declarada** | |

Un F1-macro de 0,91 con una relación casi determinista entre entradas y salida es la **firma de un
dataset generado por fórmula** con un poco de ruido encima, no de mediciones de campo. En un problema
real de desperdicio alimentario influyen decenas de variables no observadas y nadie llega a 0,91.

Conclusión: `food_wastage_data.csv` **daba números mucho más bonitos y mucho menos defendibles**. Lo
conservamos en `legacy/` como registro del proceso, pero la entrega usa FIKWaste porque es
verificable, tiene DOI, licencia explícita y mediciones reales — aunque su techo de desempeño sea
menos de la mitad.

> El script de la auditoría es reproducible con las mismas librerías del notebook; los números de
> arriba salen de `legacy/food_wastage_data.csv` con `random_state=42`.

---

## 2 · Contenido crudo

```
data/raw/FIKWaste/
├── deployments.csv                     # metadatos de las 3 cocinas
├── Kitchen 1/  Paper · Plastic · Undifferentiated
├── Kitchen 2/  Glass · Paper · Plastic · Undifferentiated
└── Kitchen 3/  Glass · Paper · Plastic · Undifferentiated
       └── cada contenedor: measurements.csv + labels.csv
```

**86.044 mediciones** y **106 eventos de vaciado etiquetados**, repartidos en **11 series**
cocina×contenedor (la Cocina 1 no tiene contenedor de vidrio monitoreado).

| Cocina | Servicio | Área | Aforo | Contenedores | Mediciones | Periodo |
|---|---|---:|---:|---|---:|---|
| 1 | Cena | 58,15 m² | 50 | Paper, Plastic, Undiff. | 58.647 | 6 feb – 3 mar 2019 |
| 2 | Cena | 25,52 m² | 50 | + Glass | 11.281 | 12 mar – 2 abr 2019 |
| 3 | Desayuno y cena | 35,23 m² | 40 | + Glass | 16.116 | 16 abr – 15 may 2019 |

### Esquema

**`measurements.csv`** — `timestamp` (datetime) · `distance` (cm, sensor→residuo) · `volume` (% de llenado)
**`labels.csv`** — `timestamp` · `volume` (% en ese momento) · `source` (`V` = vídeo, `H` = humano)
**`deployments.csv`** — `ID`, `service`, `area` (m²), `capacity` (comensales), `has_*`/`*_volume` (m³) por fracción, `start`, `end`

`volume = (A_base × H_sensor) / (A_base × H_bin) × 100`

---

## 3 · Preprocesamiento

Todo el proceso está en el notebook `notebooks/M1_finetuning_LoRA_FIKWaste.ipynb`, secciones 2–7,
y produce `data/processed/fikwaste_hourly_windows.csv` (3.306 filas × 24 columnas).

**Pasos:**

1. **Descarga** desde OSF (o uso de la copia local del repo).
2. **Agregación a ventanas de 1 hora** por cocina y contenedor: media, máximo, mínimo del llenado y
   número de lecturas. *Por qué 1 h:* a 5 min la señal es ruido de sensor; a 1 día quedan solo 249
   filas.
3. **Features de historia** (todas miran hacia atrás): `delta_1h`, `delta_3h`, `mean_3h`.
4. **`hours_since_disposal`** vía `merge_asof` contra `labels.csv`.
5. **Contexto temporal y metadatos** de la cocina.
6. **Serialización a texto en inglés** con plantilla fija y determinista.
7. **Split temporal** por día dentro de cada cocina.

---

## 4 · Tarea

**Clasificación multiclase de series temporales, reformulada como clasificación de texto.**

- **Input:** descripción en inglés del estado de un contenedor en la hora `t` — tipo de cocina y de
  contenedor, día y hora, nivel de llenado actual, cambio en la última hora y en las últimas tres,
  y horas desde el último vaciado registrado.
- **Output:** el nivel de residuo que acumulará **entre `t` y `t+1`**.

```
incremento = max(0, volume(t+1) − volume(t))
```

| Clase | Incremento | Lectura operativa |
|---|---|---|
| `LOW` | ≤ 1 punto | el contenedor prácticamente no se movió |
| `MEDIUM` | 1 – 8 puntos | acumulación normal de servicio |
| `HIGH` | > 8 puntos | pico de generación; puede tocar vaciar pronto |

Los cortes son **fijos e interpretables**, no terciles: al personal de cocina le importa el umbral
real de llenado, no el cuantil.

### Chequeo de leakage

`next_volume` e `increment_next_h` son el target y **no entran** en el texto de entrada. El notebook
lo verifica con un `assert` explícito (sección 5.1).

---

## 5 · Tamaño y split

| | Filas | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|---:|
| **Train** | 2.738 | 68,8 % | 19,6 % | 11,6 % |
| **Validation** | 568 | 66,9 % | 22,4 % | 10,7 % |
| **Total** | **3.306** | 68,5 % | 20,1 % | 11,4 % |

**El split es temporal, no aleatorio.** El primer 80 % de los días de cada cocina va a train, el 20 %
final a validation:

| Cocina | Días | Corte | Train | Val |
|---|---:|---|---:|---:|
| 1 | 25 | 2019-02-27 | 813 | 183 |
| 2 | 22 | 2019-03-29 | 774 | 146 |
| 3 | 29 | 2019-05-09 | 1.151 | 239 |

*Por qué:* un split aleatorio **fuga información** en series temporales. Las ventanas de las 14:00 y
las 15:00 del mismo día comparten casi todo (`mean_3h` y `delta_3h` se solapan); si una cae en train
y la otra en validation, el modelo interpola en vez de generalizar y la métrica sale inflada. El
split temporal valida sobre días nunca vistos, que es el escenario de uso real.

---

## 6 · Idioma

- **Datos crudos:** numéricos, sin idioma. Nombres de columna y de carpeta en **inglés**.
- **Texto de entrada al modelo:** **inglés**, generado por plantilla. Se eligió inglés porque el
  modelo base (`distilbert-base-uncased`) está preentrenado solo en inglés; usar español lo
  penalizaría sin ninguna ventaja, dado que el texto es sintético y no hay hablantes involucrados.
- **Documentación y notebook:** español.

---

## 7 · Sesgos y limitaciones conocidas

> Sección obligatoria de la rúbrica. Estas son las razones por las que el modelo probablemente falla,
> y por las que no debería usarse fuera de este contexto.

1. **Muestra minúscula y no representativa.** Tres cocinas de un mismo país, cuatro semanas de 2019
   cada una. Dos sirven solo cenas. No hay comida rápida, ni cadenas, ni catering, ni cocinas
   industriales. **Cualquier conclusión sobre "restaurantes" en general sería injustificada.**

2. **Frecuencia de muestreo desigual.** La Cocina 1 mide cada minuto y 24 h al día; las Cocinas 2 y 3
   cada cinco minutos y solo en horario laboral (para ahorrar batería). Esto hace que la Cocina 1
   aporte **más ventanas de las que le corresponden por días monitoreados**, y con ellas su patrón
   particular de generación de residuos. El modelo está sesgado hacia esa cocina.

3. **Es volumen (%), no peso.** El sensor mide distancia. Un 10 % de llenado de plástico y un 10 % de
   residuo orgánico **no son la misma cantidad de desperdicio en kg**. Sin densidades por fracción no
   se puede convertir a masa, así que los resultados no son comparables con estudios que reportan
   toneladas.

4. **El cero no es cero.** El propio paper lo documenta: las bolsas vacías no se estiran del todo y el
   peso del papel y el plástico impide que caigan al fondo, así que un contenedor recién vaciado puede
   marcar 20–30 %. El offset es **distinto en cada contenedor**, lo que introduce un sesgo sistemático
   por contenedor que el modelo puede estar aprendiendo como si fuera señal.

5. **Labels de vaciado incompletos por diseño.** Solo se conservaron los eventos confirmados por al
   menos 2 de los 3 anotadores, y el vídeo de referencia no estuvo disponible todo el tiempo. Hay
   vaciados reales sin etiquetar → `hours_since_disposal` es ruidoso y a veces **sobreestima** el
   tiempo transcurrido. Un 9,3 % de las ventanas no tiene ningún vaciado previo registrado.

6. **Bajadas de volumen que no son vaciados.** El personal reacomoda las bolsas, y eso produce
   descensos de nivel que no corresponden a que se haya sacado la basura. Recortamos el incremento en
   0 para no tratarlos como "residuo negativo", pero contaminan `delta_1h` y `delta_3h`.

7. **`Undifferentiated` ≈ residuo alimentario, pero no exactamente.** Es la fracción no reciclable.
   Contiene sobre todo comida, pero también servilletas, envoltorios y otros no-alimentos. **No es una
   medida limpia de food waste.**

8. **Sin contexto de negocio.** No hay número de comensales por servicio, ni menú, ni facturación, ni
   clima. Faltan justamente las variables que más explicarían la generación de residuos, así que el
   techo de desempeño alcanzable con este dataset es bajo por construcción.

9. **El texto de entrada es sintético.** No hay lenguaje natural en el dominio: la descripción la
   genera una plantilla fija. El modelo no está aprendiendo "lenguaje del sector de la restauración",
   está aprendiendo a leer números embebidos en una frase constante. Es la limitación de fondo de todo
   el planteamiento y se discute en la sección 17 del notebook.

### Consideración ética

Los datos **no contienen información personal**: son lecturas de distancia de un sensor sobre un
contenedor. No hay clientes, ni empleados, ni imágenes identificables. Los vídeos que se usaron para
etiquetar los vaciados **no forman parte del dataset publicado**; los autores solo liberaron los
timestamps derivados. Los restaurantes están anonimizados como "Kitchen 1/2/3".

El riesgo ético relevante no está en la privacidad sino en el **uso indebido por sobregeneralización**:
presentar conclusiones sacadas de 3 cocinas portuguesas de 2019 como si aplicaran al sector. Este
documento existe en buena parte para dejar constancia de que no aplican.

---

## 8 · Cita

```bibtex
@article{pereira2021fikwaste,
  title   = {FIKWaste: A Waste Generation Dataset from Three Restaurant Kitchens in Portugal},
  author  = {Pereira, Lucas and Aguiar, Vitor and Vasconcelos, Filipe},
  journal = {Data},
  volume  = {6},
  number  = {3},
  pages   = {25},
  year    = {2021},
  doi     = {10.3390/data6030025}
}
```

### Datasets hermanos (mismas 3 cocinas)

- **FIKElectricity** — consumo eléctrico · https://doi.org/10.1038/s41597-023-02698-8
- **FIKWater** — consumo de agua · https://doi.org/10.3390/data6030026 · datos: https://osf.io/7bz2m/

Permiten cruzar generación de residuos con consumo energético e hídrico por cocina; queda como línea
abierta para los módulos siguientes.
