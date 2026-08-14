# Descripción del dataset — FIKWaste

Este documento corresponde al **punto 2 de la entrega M1**: fuente, tamaño, idioma, licencia, tarea y limitaciones del dataset.

## 1. Fuente

Usamos **FIKWaste (A Waste Generation Dataset from Three Restaurant Kitchens in Portugal)**, publicado en 2021 por Pereira, Aguiar y Vasconcelos.

- Fuente: https://osf.io/tyaj6/
- Paper: Pereira, L.; Aguiar, V.; Vasconcelos, F. *FIKWaste: A Waste Generation Dataset from Three Restaurant Kitchens in Portugal*. Data, 2021, 6, 25.
- DOI: https://doi.org/10.3390/data6030025
- Licencia: **CC BY 4.0**
- Datos usados: `data/raw/FIKWaste/`

El dataset contiene mediciones reales tomadas con sensores ultrasónicos instalados en contenedores de residuos de tres cocinas de restaurante en Portugal durante 2019. El sensor registra la distancia hasta la superficie del residuo y a partir de ella se obtiene un porcentaje de llenado.

Antes de usar FIKWaste también revisamos el dataset que estaba en `legacy/food_wastage_data.csv`. Se dejó fuera de la entrega porque no tenía una fuente primaria clara ni una licencia explícita y presentaba una relación muy fuerte entre las variables y el target. FIKWaste fue una opción más fácil de justificar porque tiene una fuente publicada, DOI y licencia.

## 2. Tamaño y estructura

El dataset contiene **86.044 mediciones** y **106 eventos de vaciado etiquetados**, distribuidos en **11 combinaciones cocina × tipo de contenedor**.

| Cocina | Contenedores | Mediciones | Periodo aproximado |
|---|---|---:|---|
| Kitchen 1 | Paper, Plastic, Undifferentiated | 58.647 | feb. – mar. 2019 |
| Kitchen 2 | Glass, Paper, Plastic, Undifferentiated | 11.281 | mar. – abr. 2019 |
| Kitchen 3 | Glass, Paper, Plastic, Undifferentiated | 16.116 | abr. – may. 2019 |

Kitchen 1 no tiene un contenedor de vidrio monitoreado.

Los archivos principales son:

- `measurements.csv`: timestamp, distancia del sensor y porcentaje de llenado.
- `labels.csv`: timestamps de eventos de vaciado y su origen.
- `deployments.csv`: información de cada cocina y de los contenedores.

Después del preprocesamiento se obtienen **3.306 ventanas de una hora**:

| Split | Filas | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|---:|
| Train | 2.738 | 68,8 % | 19,6 % | 11,6 % |
| Validation | 568 | 66,9 % | 22,4 % | 10,7 % |
| Total | 3.306 | 68,5 % | 20,1 % | 11,4 % |

El split es temporal. Para cada cocina, el primer 80 % de los días se usa para entrenamiento y el 20 % final para validación. Así evitamos mezclar horas muy cercanas del mismo día entre train y validation.

## 3. Idioma y licencia

Los datos originales son principalmente numéricos y no contienen texto natural. La entrada del modelo se construye en **inglés** a partir de una plantilla. Esto se hizo porque el modelo escogido, `distilbert-base-uncased`, está preentrenado en inglés.

El dataset está publicado bajo **CC BY 4.0**, por lo que se puede reutilizar con la atribución correspondiente.

## 4. Tarea

La tarea es una **clasificación multiclase**.

Para cada ventana de una hora, el modelo recibe información disponible hasta la hora `t`:

- cocina;
- tipo de contenedor;
- nivel de llenado;
- cambio de llenado en la última hora;
- cambio de llenado en las últimas tres horas;
- promedio reciente;
- tiempo desde el último vaciado etiquetado;
- hora y día de la semana;
- algunos metadatos de la cocina y del contenedor.

El target corresponde al aumento de llenado entre `t` y `t+1`:

```text
incremento = max(0, volume(t+1) - volume(t))
```

Se transforma en tres clases:

| Clase | Aumento de llenado | Interpretación |
|---|---:|---|
| `LOW` | ≤ 1 punto | Poco o ningún aumento |
| `MEDIUM` | > 1 y ≤ 8 puntos | Aumento moderado |
| `HIGH` | > 8 puntos | Aumento alto |

Los valores de la siguiente hora no se incluyen en el texto de entrada. El notebook tiene un chequeo explícito para evitar usar las columnas del target como features.

## 5. Preprocesamiento

El proceso se encuentra en el notebook de la entrega. En resumen:

1. Se cargan las mediciones, labels y metadatos.
2. Las mediciones se agrupan en ventanas de una hora por cocina y contenedor.
3. Se calculan cambios y promedios usando información anterior a la hora que se quiere predecir.
4. Se incorpora el tiempo desde el último vaciado registrado.
5. Se agregan variables temporales y metadatos de la cocina.
6. Las variables se convierten en una descripción en inglés para poder usarlas con DistilBERT.
7. Se genera el split temporal de entrenamiento y validación.

En la versión v2 se comparan dos formas de construir el texto: una mantiene los valores numéricos y la otra los convierte a categorías lingüísticas como "nearly full" o "rose sharply".

## 6. Limitaciones y posibles sesgos

El dataset es útil para probar el modelo, pero tiene varias limitaciones importantes:

1. **Son solo tres cocinas de Portugal.** Los resultados no permiten afirmar que el modelo funcione igual en otros restaurantes o países.
2. **El periodo es corto.** Los datos corresponden a semanas de 2019, por lo que no representan cambios de temporada ni diferentes años.
3. **El muestreo no es igual.** Kitchen 1 registra datos cada minuto, mientras que Kitchens 2 y 3 lo hacen cada cinco minutos y en horarios diferentes.
4. **Se mide volumen, no peso.** Un porcentaje de llenado no equivale directamente a una cantidad de desperdicio en kilogramos.
5. **Un descenso de volumen no siempre significa que se vació el contenedor.** Puede deberse a que se reacomodaron las bolsas.
6. **Los eventos de vaciado no están completos.** Algunos eventos reales pueden no tener una etiqueta, por lo que `hours_since_disposal` puede contener ruido.
7. **`Undifferentiated` no es food waste puro.** Es la fracción no diferenciada y puede contener otros residuos además de comida.
8. **La entrada del modelo es texto generado por plantilla.** No es texto escrito por personas. Por eso el experimento evalúa la capacidad de un encoder para trabajar con una representación textual de datos numéricos, no la comprensión de lenguaje natural sobre restaurantes.
9. **Hay desbalance de clases.** La mayoría de las ventanas son `LOW`, mientras que `HIGH` es la clase menos frecuente. Por eso se usa F1-macro como métrica principal.

## 7. Consideración ética

Los datos publicados no contienen información personal de clientes o empleados. El principal riesgo en este proyecto es otro: interpretar los resultados como si representaran a todos los restaurantes. El modelo debe entenderse como un experimento sobre las tres cocinas incluidas en FIKWaste.

## 8. Referencia

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
