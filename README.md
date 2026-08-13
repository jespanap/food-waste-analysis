# Predicción del ritmo de generación de residuos en cocinas de restaurante

> **¿Qué hace este sistema?**
> Mira el estado del contenedor de basura de la cocina de un restaurante —cuán lleno está, cómo viene
> llenándose, qué hora y qué día es— y **anticipa cuánta basura va a acumular en la próxima hora**.
> Responde una de tres cosas: poca, media o mucha. Sirve para que el personal sepa cuándo va a tocar
> vaciarlo antes de que se desborde, y para que el restaurante detecte a qué horas y en qué días está
> generando más desperdicio.

**SI4006 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT**
Entrega **M1** — Módulo 1: arquitectura transformer y fine-tuning eficiente

<!-- TODO: renombrar el repositorio con los nombres de los integrantes del equipo (lo pide la asignación) -->
**Equipo:** _(completar)_

---

## Cómo correrlo

1. Abrir **`notebooks/M1_finetuning_LoRA_FIKWaste_v2.ipynb`** en Google Colab.
2. `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`.
3. `Entorno de ejecución → Ejecutar todo`. Tarda ~12 min (dos entrenamientos).

El notebook **descarga el dataset por sí solo** desde OSF, así que no hay que subir archivos ni
montar Drive. Si clonas el repo, usa la copia local de `data/raw/FIKWaste/` y se salta la descarga.
Semilla fijada en `42`. Corre completo en Colab gratuito.

> El **v1** (`M1_finetuning_LoRA_FIKWaste.ipynb`) también corre y se conserva: es la primera
> iteración, y el v2 lo usa como punto de comparación. Ver [Resultados](#resultados).

---

## Estructura del repositorio

```
.
├── README.md                                     ← este archivo (reporte de resultados)
├── docs/
│   └── DATASET.md                                ← fuente, tamaño, licencia, sesgos del dataset
├── notebooks/
│   ├── M1_finetuning_LoRA_FIKWaste.ipynb         ← v1: primera iteración
│   └── M1_finetuning_LoRA_FIKWaste_v2.ipynb      ← v2: ablación representación × pesos  ⭐ entrega
├── data/
│   ├── raw/FIKWaste/                             ← dataset crudo: 3 cocinas, 86.044 mediciones (CC BY 4.0)
│   │   ├── deployments.csv
│   │   ├── FIKWaste_paper_Data2021.pdf           ← paper con la metodología de captura
│   │   └── Kitchen 1|2|3/<fracción>/{measurements,labels}.csv
│   └── processed/
│       ├── fikwaste_hourly_windows.csv           ← salida del preprocesamiento v1
│       └── fikwaste_hourly_windows_v2.csv        ← v2: añade las dos columnas de texto
└── legacy/                                       ← versión anterior del proyecto, como registro
    ├── NOTEBOOK FOODWASTE.ipynb
    └── food_wastage_data.csv
```

---

## El dataset

**FIKWaste** — datos reales de sensores ultrasónicos instalados en los contenedores de **3 cocinas de
restaurante en Portugal** (2019). 86.044 mediciones, 106 eventos de vaciado etiquetados, 11 series
cocina×contenedor.

- **Fuente:** https://osf.io/tyaj6/ · **Licencia:** CC BY 4.0
- **Paper:** Pereira, Aguiar & Vasconcelos, *Data* 2021, 6, 25 · https://doi.org/10.3390/data6030025
- **Descripción completa, split y sesgos:** [`docs/DATASET.md`](docs/DATASET.md)

Tras el preprocesamiento quedan **3.306 ventanas horarias**: 2.738 de entrenamiento y 568 de
validación, con un **split temporal** (el 20 % final de días de cada cocina se reserva para validar).

> **Nota:** la versión anterior de este proyecto usaba un CSV de Kaggle sin fuente primaria y con
> toda probabilidad sintético. Se conserva en `legacy/` como registro del proceso, pero la entrega usa
> FIKWaste porque es verificable, tiene DOI y licencia explícita.

---

## Decisiones técnicas

### Familia y modelo base

| | |
|---|---|
| **Familia** | Encoder-only |
| **Modelo base** | `distilbert-base-uncased` (66M parámetros) |

**Por qué encoder-only.** El objetivo de preentrenamiento debe casar con la tarea. Los encoder-only se
preentrenan con *masked language modeling*: atención bidireccional, ven toda la secuencia a la vez, y
su fuerte es **clasificar y extraer**. Nuestra entrada es una descripción completa que hay que leer
entera antes de asignarle una categoría — no generamos ni transformamos texto. Encaja.

**Por qué DistilBERT y no BERT-base.** DistilBERT es un destilado de BERT-base: ~40 % menos parámetros
y ~60 % más rápido, reteniendo ~97 % del desempeño en GLUE. Con ~2.700 ejemplos de entrenamiento el
cuello de botella son los datos, no la capacidad del modelo; gastar el doble de cómputo en BERT-base
no compraría nada.

**Lo que vimos en la tokenización, y lo que hicimos con ello.** Siguiendo el consejo del Lab A,
revisamos cómo trata el tokenizer nuestro vocabulario. `undifferentiated` se parte en varias piezas de
WordPiece, pero aparece en todas las filas del mismo contenedor y el modelo lo aprende como patrón
fijo. El problema real son los **números**: WordPiece fragmenta `57.0` → `['57','.','0']` y
`64.9` → `['64','.','9']`, dos secuencias sin nada en común pese a distar 8 puntos. **No preserva
magnitud**, que es la única información que llevan nuestras features.

Esto empezó como una observación en el v1 y terminó siendo **el hallazgo principal del proyecto**:
en el v2 lo convertimos en un experimento y resultó ser el factor dominante. Ver
[Resultados](#resultados).

### Las dos representaciones de texto

FIKWaste es señal de sensor, no texto, así que serializamos cada ventana horaria a una descripción en
inglés. El v2 compara **dos formas de hacerlo** con exactamente las mismas features:

| | Ejemplo |
|---|---|
| **A · numérica** (la del v1) | `The container is currently 86.2 percent full. Over the last hour the fill level changed by 16.4 points.` |
| **B · bucketizada** (nueva) | `The container is nearly full. Over the last hour the fill level rose sharply.` |

La versión bucketizada convierte cada magnitud en una **expresión léxica ordinal** cuyos cortes se
calibraron sobre los percentiles reales del dataset, no a ojo. La apuesta: `about half full` y
`more than half full` comparten tokens y son expresiones que DistilBERT vio millones de veces en el
preentrenamiento, con su orden semántico ya codificado — al contrario que `57.0` y `64.9`.

**Lo que se pierde:** resolución. `57 %` y `64 %` caen en el mismo bucket. Resultó ser un buen
intercambio, pero limita el techo alcanzable.

### Hiperparámetros de LoRA

| Hiperparámetro | Valor | Por qué |
|---|---|---|
| `r` (rank) | `8` | ~2.700 ejemplos y texto de plantilla fija (poca variación léxica). `r=8` da capacidad suficiente sin sobreajustar; `r=16+` con este volumen memoriza. |
| `lora_alpha` | `16` | Regla estándar `alpha ≈ 2·r`; el factor `alpha/r = 2` mantiene estable la magnitud de la actualización. |
| `lora_dropout` | `0.10` | Dataset pequeño y muy repetitivo → regularización explícita. |
| `target_modules` | `["q_lin", "v_lin"]` | Proyecciones *query* y *value* de la atención en DistilBERT. Adaptar Q y V es el estándar del paper de LoRA: es donde se decide *a qué atiende* el modelo, que es lo que cambia entre dominios. |
| `modules_to_save` | `["pre_classifier", "classifier"]` | **Crítico.** La cabeza de 3 clases se inicializa aleatoriamente; sin marcarla como entrenable, LoRA ajustaría el cuerpo contra una cabeza al azar. |

Entrenamiento: `lr=2e-4` (alto para full fine-tuning, normal en LoRA), 6 épocas, batch 16,
`load_best_model_at_end` por `f1_macro`. **LoRA no cambió entre v1 y v2** — es lo que mantuvimos fijo
para que la ablación significara algo.

### Pesos de clase

Con 67 / 21 / 11, la salida degenerada "responder siempre `LOW`" es un mínimo local muy cómodo, así que
ponderamos la *cross-entropy*. El v2 también compara dos formas:

| | Fórmula | Razón `HIGH`:`LOW` | Resultado medido |
|---|---|---:|---|
| **v1** | inverso de frecuencia | 5.95:1 | Sobrepredice minorías: 152 alarmas de `HIGH` para 61 picos reales |
| **v2** | raíz del inverso, renormalizada | 2.44:1 | Corrige la accuracy, pero con texto numérico se pasa al otro lado |

La conclusión del experimento es que **el problema no eran los pesos**: con la representación numérica
no hay valor que funcione (con 6:1 sobrepredice, con 2.4:1 ignora las minorías). Lo que desbloqueó el
modelo fue cambiar la representación.

### Métrica principal: F1-macro

Promedia el F1 de las tres clases con el mismo peso. Es la correcta aquí porque las clases están
desbalanceadas (67 / 21 / 11) y **la clase operativamente valiosa es `HIGH`** —la que avisa de un pico
de generación— que además es la minoritaria. Con accuracy, acertar `HIGH` casi no mueve el número;
con F1-macro pesa un tercio.

### Baselines

1. **Clase mayoritaria** — el piso absoluto. Sirve sobre todo para exponer el desbalance: responder
   siempre `LOW` da ~0,67 de accuracy y parece decente, pero su F1-macro es ~0,27. Es el argumento de
   por qué la métrica principal no puede ser accuracy.
2. **DistilBERT sin fine-tuning (zero-shot)** — el baseline recomendado por la asignación. Usa el
   objetivo original de MLM: se le pide completar `The waste generation level is [MASK].` y se comparan
   las probabilidades de `low`, `medium` y `high`. Mide qué sabe el modelo base *antes* de afinarlo, así
   que el delta contra él es limpiamente atribuible a LoRA.

---

## Resultados

Ejecución del 13 de agosto de 2026 en Colab (Tesla T4, `transformers` 5.13.1, `peft` 0.20.0).
Validación = últimos ~20 % de días de cada cocina (568 ventanas).

El proyecto se hizo en **dos iteraciones**. La v1 dejó dos hipótesis sin medir; la v2 las convirtió en
una **ablación de dos factores** (representación × pesos de clase), cambiando un solo factor por
corrida para poder atribuir la diferencia.

| Modelo | Accuracy | Precision-macro | Recall-macro | **F1-macro** |
|---|---:|---:|---:|---:|
| Baseline: clase mayoritaria (`LOW`) | 0.6690 | 0.2230 | 0.3333 | 0.2672 |
| Baseline: DistilBERT zero-shot | 0.6690 | 0.2230 | 0.3333 | 0.2672 |
| v1 · numérico + pesos duros (6:1) | 0.4137 | 0.4021 | 0.4434 | 0.3747 |
| Run A · numérico + pesos suaves (2.4:1) | 0.6496 | 0.4556 | 0.3894 | 0.3859 |
| Run B · bucketizado + pesos suaves (2.4:1) · semilla 42 | 0.6479 | 0.5059 | 0.5056 | 0.5036 |
| **Run B · promedio sobre 3 semillas** ⭐ | **0.6508 ± 0.0027** | — | — | **0.4784 ± 0.0271** |
| _[referencia]_ HistGradientBoosting tabular | 0.6655 | 0.5021 | 0.4283 | 0.4335 |

**El resultado del proyecto es `F1-macro = 0.478 ± 0.027`**, no el 0.5036 de la semilla 42 — que
resultó ser la mejor de las tres. Detalle abajo.

| Semilla | Accuracy | F1-macro |
|---|---:|---:|
| 42 | 0.6479 | 0.5036 |
| 202 | 0.6514 | 0.4498 |
| 1337 | 0.6532 | 0.4820 |

**Efectos aislados (F1-macro):**

| Factor | Comparación | Δ |
|---|---|---:|
| Pesos de clase (6:1 → 2.4:1) | A − v1 | **+0.0112** |
| Representación (numérica → bucketizada) | B̄ − A | **+0.0925** |
| Mejor modelo vs baselines | B̄ − zero-shot | **+0.2112** |
| Mejor modelo vs árboles | B̄ − HistGB | **+0.0449** |

*(B̄ = promedio de las 3 semillas.)*

Por clase (Run B, el modelo de la entrega):

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| `LOW` | 0.760 | 0.800 | 0.779 | 380 |
| `MEDIUM` | 0.390 | 0.307 | 0.344 | 127 |
| `HIGH` | 0.368 | 0.410 | 0.388 | 61 |

LoRA entrenó **740.355 de 67.696.134 parámetros (1.09 %)**; el adaptador pesa 3.7 MB.

### Lectura honesta

**La representación pesó ocho veces más que los pesos de clase.** Bucketizar los números dio +0.0925
de F1-macro (promediando semillas); suavizar los pesos, +0.0112. Y aguanta: incluso con la **peor** de
las tres semillas el efecto de la representación sigue siendo +0.0639, más de cinco veces el de los
pesos. Esta conclusión no depende de la suerte. Y son efectos de distinta naturaleza: los pesos casi no
movieron el F1 pero **arreglaron la accuracy** (0.4137 → 0.6496), es decir, dejaron de destrozar la
clase mayoritaria sin ganar nada real en las minorías. De hecho sobrecorrigieron hacia el otro lado —
el Run A emitió 500 `LOW`, 41 `MEDIUM` y 27 `HIGH` cuando la realidad era 380 / 127 / 61. **Con la
representación numérica no hay un valor de pesos que funcione:** con 6:1 sobrepredice minorías, con
2.4:1 las ignora.

**La hipótesis de la v1 quedó confirmada.** La v1 propuso que WordPiece destruye la magnitud numérica
(`57.0` → `['57','.','0']`, `64.9` → `['64','.','9']`: dos tokens sin nada en común pese a distar 8
puntos). Al reemplazar los números por expresiones ordinales que el modelo ya conoce del
preentrenamiento (`about half full`, `rose sharply`), el F1-macro sube +0.0925 **sin tocar nada más**.
Se ve también en la pérdida, que es donde se nota si el modelo aprende o solo reparte predicciones: la
v1 se quedó en 1.0615 (pegada a `ln(3) = 1.0986`, o sea al azar), el Run A bajó a 0.9672 y el **Run B a
0.8841**, la única de las tres que se despega de verdad. Y de paso usa **28 % menos tokens** (76.4 vs
105.8 de media): rinde mejor y es más barata.

**¿Le ganamos al árbol? De forma consistente, pero no está demostrado.** Es la afirmación más fuerte
del trabajo, así que conviene ser preciso. A favor: el punto estimado favorece al transformer (0.4784
vs 0.4335, **+0.0449**) y **las tres semillas superan al árbol**, incluso la peor (0.4498, +0.0163). En
contra: con n = 3 el error estándar es 0.0156 y el **IC 95 % es [0.411, 0.546]**, un intervalo que
**contiene el 0.4335 del árbol**; además el árbol es determinista, así que solo medimos la varianza de
un lado. La redacción defendible es: *el modelo afinado quedó consistentemente por delante en las tres
corridas, con una ventaja media de +0.045; con solo tres semillas la diferencia no alcanza
significancia estadística, pero la dirección es estable.*

**Lo que sí queda firmemente establecido** es lo otro. La v1 concluyó que "para datos de sensor un
encoder de texto es la herramienta equivocada", con 0.3747 frente a 0.4335 del árbol. Esa conclusión
**está refutada**: la distancia entre 0.3747 y 0.4784 es de casi 4 desviaciones estándar. La lectura
correcta no es "el transformer es mejor que los árboles", sino: **el problema nunca fue el modelo, era
la interfaz entre el dato y el modelo.** Un encoder puede competir con un árbol en datos de sensor *si
se le habla en su idioma*, y su idioma son palabras con orden semántico, no dígitos que su tokenizer
despedaza.

**Traducido a operación:** contando solo las alarmas de `HIGH`, la v1 emitía 152 para 61 picos reales
(1 de cada 5.1 acertaba); el Run A se callaba casi siempre (27 alarmas, detectaba 9 de 61); el Run B
emite 68 y acierta 25 — **1 de cada 2.7**. Sigue sin ser un sistema de producción, pero ya no es
absurdo. El reparto de predicciones del Run B (razones 1.05 / 0.79 / 1.11 respecto a la realidad) está
prácticamente calibrado.

**Dónde falla ahora:** `MEDIUM` es la peor clase (F1 0.344), desbancando a `HIGH`. Tiene sentido — es
la banda intermedia, la única cuyos dos límites son convencionales. `LOW` es "no pasó nada", `HIGH` es
"pasó algo grande", `MEDIUM` es todo lo demás. El otro modo de fallo aparece en las ventanas **sin
historia disponible**: sin `delta_1h` ni `delta_3h`, el modelo se queda con el nivel y la hora, y
dispara.

**Lo que enseñó el estudio de semillas.** Repetir la mejor configuración con 42, 202 y 1337 dio
0.5036 / 0.4498 / 0.4820: **la semilla 42 era la mejor de las tres**, así que reportar su 0.5036 habría
inflado la cifra en ~0.025. Por eso el resultado del proyecto es 0.478 ± 0.027.

Y salió un hallazgo secundario bastante informativo: la **accuracy es diez veces más estable que el
F1-macro** (±0.0027 vs ±0.0271). La explicación es clara — el modelo acierta la clase mayoritaria de
forma fiable corrida tras corrida, y **toda la varianza está en las minorías**, que es justo donde el
F1-macro pone su peso. Con 61 ejemplos de `HIGH` en validación, que cambien de bando cinco o seis
predicciones mueve el macro-promedio varios puntos. Es el precio de elegir la métrica correcta: mide lo
que importa, pero con este tamaño de muestra mide con ruido.

Queda pendiente lo más importante metodológicamente: **separar un conjunto de test de verdad**. Ahora
seleccionamos la mejor época sobre validación y reportamos sobre validación, lo que infla las tres
cifras por igual.

Detalle completo en la sección 20 del notebook v2.

---

## Limitaciones del dataset

Resumen; el detalle está en [`docs/DATASET.md`](docs/DATASET.md#7--sesgos-y-limitaciones-conocidas).

- **Muestra minúscula:** 3 cocinas de un país, 4 semanas de 2019. No generaliza al sector.
- **Muestreo desigual:** la Cocina 1 mide cada minuto y las otras cada 5 → está sobrerrepresentada.
- **Es volumen (%), no kg:** 10 % de plástico y 10 % de orgánico no son el mismo desperdicio.
- **El cero no es cero:** las bolsas vacías no se estiran; un contenedor recién vaciado marca 20–30 %,
  con un offset distinto en cada contenedor.
- **Labels de vaciado incompletos por diseño** → `hours_since_disposal` es ruidoso.
- **`Undifferentiated` ≠ food waste puro:** es la fracción no reciclable, incluye no-alimentos.
- **El texto de entrada es sintético:** no hay lenguaje natural en el dominio.

---

## Cita del dataset

Pereira, L.; Aguiar, V.; Vasconcelos, F. *FIKWaste: A Waste Generation Dataset from Three Restaurant
Kitchens in Portugal.* **Data** 2021, 6, 25. https://doi.org/10.3390/data6030025 — CC BY 4.0
