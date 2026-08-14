# Resultados — FIKWaste + DistilBERT con LoRA

Este documento corresponde al **punto 3 de la entrega M1**. Resume las decisiones del modelo, los baselines y los resultados obtenidos en la validación.

## 1. Modelo elegido

Usamos **DistilBERT (`distilbert-base-uncased`)**, de la familia **encoder-only**.

La tarea es clasificar una descripción del estado actual de un contenedor en `LOW`, `MEDIUM` o `HIGH`. No necesitamos generar texto: necesitamos leer la entrada completa y asignarle una categoría. Por eso un encoder es una elección más directa que un decoder o un encoder-decoder.

DistilBERT se escogió porque es más pequeño que BERT-base y permite trabajar con una GPU T4 y un dataset relativamente pequeño.

## 2. Fine-tuning

El ajuste se hizo con **LoRA**, sin hacer full fine-tuning.

| Parámetro | Valor |
|---|---:|
| `r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.10 |
| `target_modules` | `q_lin`, `v_lin` |
| `modules_to_save` | `pre_classifier`, `classifier` |
| Learning rate | `2e-4` |
| Épocas | 6 |
| Batch de entrenamiento | 16 |
| Métrica para escoger el mejor modelo | F1-macro |

Se entrenan las capas de LoRA y la cabeza de clasificación, mientras que la mayor parte de DistilBERT permanece congelada.

## 3. Baselines

Se usaron dos baselines:

- **Clase mayoritaria:** siempre predice `LOW`. Sirve como referencia mínima y permite ver el efecto del desbalance.
- **DistilBERT zero-shot:** se usa el modelo base sin fine-tuning y se comparan las probabilidades de `low`, `medium` y `high` en una posición `[MASK]`.

También se entrenó un `HistGradientBoostingClassifier` con las mismas variables numéricas. Este último no es el baseline exigido por la asignación, sino una referencia para saber qué ocurre si tratamos los datos como tabulares y no como texto.

## 4. Resultados

Todos los modelos se evaluaron sobre las mismas **568 ventanas de validation**.

| Modelo | Accuracy | Precision-macro | Recall-macro | **F1-macro** |
|---|---:|---:|---:|---:|
| Clase mayoritaria (`LOW`) | 0.6690 | 0.2230 | 0.3333 | 0.2672 |
| DistilBERT zero-shot | 0.6690 | 0.2230 | 0.3333 | 0.2672 |
| v1: texto numérico + pesos duros | 0.4137 | 0.4021 | 0.4434 | 0.3747 |
| Run A: texto numérico + pesos suaves | 0.6496 | 0.4556 | 0.3894 | 0.3859 |
| **Run B: texto bucketizado + pesos suaves** | **0.6479** | **0.5059** | **0.5056** | **0.5036** |
| Referencia: HistGradientBoosting | 0.6655 | 0.5021 | 0.4283 | 0.4335 |

La configuración de Run B fue repetida con tres semillas:

| Semilla | Accuracy | F1-macro |
|---:|---:|---:|
| 42 | 0.6479 | 0.5036 |
| 202 | 0.6514 | 0.4498 |
| 1337 | 0.6532 | 0.4820 |
| **Media ± desviación** | **0.6508 ± 0.0027** | **0.4784 ± 0.0271** |

Para reportar el resultado general usamos la media de las tres semillas, no solamente la mejor corrida.

## 5. ¿Qué cambió en la v2?

La v2 prueba dos factores por separado.

| Factor | Comparación | Cambio en F1-macro |
|---|---|---:|
| Pesos de clase | Run A − v1 | +0.0112 |
| Representación del texto | Run B − Run A | **+0.0925** |
| Mejor configuración vs. baseline | Run B promedio − zero-shot | **+0.2112** |
| Mejor configuración vs. HistGradientBoosting | Run B promedio − HistGradientBoosting | +0.0449 |

El cambio más importante fue la forma de representar los números. En la versión numérica, valores como `57.0` y `64.9` se separan en tokens de WordPiece que no representan directamente que uno sea mayor que el otro. En la versión bucketizada se usan expresiones como `about half full` o `rose sharply`.

La mejora de la representación fue mucho mayor que la mejora obtenida al cambiar los pesos de clase.

## 6. Resultados por clase

Para la mejor corrida de Run B:

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| `LOW` | 0.760 | 0.800 | 0.779 | 380 |
| `MEDIUM` | 0.390 | 0.307 | 0.344 | 127 |
| `HIGH` | 0.368 | 0.410 | 0.388 | 61 |

La clase más difícil fue `MEDIUM`. Esto tiene sentido porque queda entre los dos extremos: `LOW` representa poco cambio y `HIGH` representa un pico, mientras que `MEDIUM` cubre una zona intermedia.

## 7. Lectura honesta

Sí hubo una mejora con respecto a los baselines. El F1-macro pasó de **0.2672** en el baseline a **0.4784 ± 0.0271** en promedio con la mejor configuración de LoRA.

Sin embargo, el resultado todavía está lejos de ser un modelo listo para producción. El dataset es pequeño y está limitado a tres cocinas. Además, el intervalo de resultados entre semillas muestra que la métrica cambia bastante cuando cambian las predicciones de las clases minoritarias.

El transformer también quedó por encima de `HistGradientBoosting` en las tres semillas del experimento, pero con solo tres corridas no se puede afirmar que la diferencia sea estadísticamente significativa.

Otro punto importante es que la validación se usa para escoger la mejor época y luego para reportar el resultado. Por eso el siguiente paso debería ser reservar un conjunto de test independiente.

## 8. Resultado que se debe presentar

Si se necesita una sola cifra para resumir el trabajo:

> **F1-macro = 0.4784 ± 0.0271**, usando DistilBERT con LoRA, texto bucketizado y pesos de clase suavizados, evaluado sobre el conjunto de validation.

La mejor corrida individual fue la semilla 42, con **F1-macro = 0.5036**, pero no se toma como resultado principal porque no representa la variación observada entre semillas.

## 9. Archivos relacionados

- Notebook principal: `notebooks/M1_finetuning_LoRA_FIKWaste_v2.ipynb`
- Dataset y metodología: `docs/DATASET.md`
- Datos procesados: `data/processed/fikwaste_hourly_windows_v2.csv`
