# Fine-tuning de DistilBERT sobre FIKWaste

Este proyecto usa FIKWaste para clasificar el aumento de residuos de un contenedor durante la siguiente hora en tres categorías: `LOW`, `MEDIUM` y `HIGH`.

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

## Requisitos

- Google Colab
- GPU T4
- Python 3
- Internet para descargar el modelo de Hugging Face y, si hace falta, FIKWaste desde OSF

Los paquetes principales se instalan desde el mismo notebook.

