# ¿Pueden los restaurantes anticipar cuánto aumentará el nivel de residuos de un contenedor?

**Este sistema ayuda a restaurantes a anticipar si el nivel de residuos de un contenedor aumentará poco, moderadamente o mucho durante la siguiente hora.** El objetivo es utilizar esta predicción como apoyo para tomar decisiones oportunas sobre la gestión de residuos.

## Fine-tuning de DistilBERT sobre FIKWaste

Usamos el dataset FIKWaste para clasificar el incremento del nivel de llenado de un contenedor durante la siguiente hora en tres categorías: `LOW`, `MEDIUM` y `HIGH`.

El modelo utiliza **DistilBERT (`distilbert-base-uncased`)**, de la familia **encoder-only**, adaptado mediante **LoRA**. Esta elección corresponde a una tarea de clasificación: el modelo recibe una descripción del estado actual del contenedor y debe asignarla a una de tres categorías, en lugar de generar texto.

## Cómo correrlo

1. Abrir `notebooks/M1_finetuning_LoRA_FIKWaste_v2.ipynb` en Google Colab.
2. Seleccionar **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU**.
3. Ejecutar las celdas en orden desde **Entorno de ejecución → Ejecutar todo**.
4. El notebook descarga FIKWaste desde OSF si no encuentra una copia local en `data/raw/FIKWaste/`.

El notebook hace el preprocesamiento, construye el texto de entrada, calcula los baselines, entrena DistilBERT con LoRA y muestra la evaluación final.

La versión `M1_finetuning_LoRA_FIKWaste.ipynb` corresponde a la primera iteración del trabajo. La entrega principal es `M1_finetuning_LoRA_FIKWaste_v2.ipynb`.

## Estructura

```text
.
├── README.md
├── docs/
│   ├── DATASET.md
│   └── RESULTADOS.md
├── notebooks/
│   ├── M1_finetuning_LoRA_FIKWaste.ipynb
│   └── M1_finetuning_LoRA_FIKWaste_v2.ipynb
├── data/
│   ├── raw/FIKWaste/
│   └── processed/
└── legacy/
```

## Dataset

La descripción completa del dataset, su fuente, tamaño, licencia, tarea y limitaciones se encuentra en [docs/DATASET.md](docs/DATASET.md).

## Resultados

Los resultados y análisis se encuentran en [docs/RESULTADOS.md](docs/RESULTADOS.md).

## Requisitos

- Google Colab
- GPU T4
- Python 3
- Internet para descargar el modelo de Hugging Face y, si hace falta, FIKWaste desde OSF

Los paquetes principales se instalan desde el mismo notebook.

