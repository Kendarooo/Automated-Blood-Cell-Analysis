# EL 5857 Aprendizaje Automático - Tarea 2
## Datos, aprendizaje supervisado y una aplicación médica

Bienvenidos al repositorio base para la Tarea 2 del curso. 

### Resumen de la Tarea

El objetivo principal de esta tarea es construir el pipeline fundacional de un hemograma automatizado a partir de imágenes de frotis de sangre periférica. Deberán implementar un flujo de trabajo de extremo a extremo que logre aislar, caracterizar y clasificar los tres tipos principales de elementos celulares: glóbulos blancos (WBC), glóbulos rojos (RBC) y plaquetas.

El sistema se compone de las siguientes etapas principales:
1. **Detección:** Uso de un modelo pre-entrenado ultrarrápido (YOLO26s) con ajuste fino para localizar las células.
2. **Extracción y aumento:** Aislamiento de las células, aplicación de aumento de datos en línea (data augmentation) para mitigar el sobreajuste, y extracción de características utilizando una red convolucional profunda congelada (ej. ResNet18).
3. **Reducción dimensional:** Aplicación de mecanismos robustos (selección univariada o regularización extrema) para manejar la alta dimensionalidad. **El uso de PCA está estrictamente prohibido**.
4. **Clasificación:** Implementación y comparación de dos enfoques:
   * Una red neuronal artificial (ANN) programada puramente en PyTorch, controlando explícitamente el ciclo de optimización y los grafos computacionales.
   * Una Máquina de Soporte Vectorial (SVM) explorando distintos kernels.
5. **Inferencia clínica:** Análisis estadístico final para contrastar proporciones celulares y emitir alertas médicas basadas en valores $p$.

Recuerden que es obligatorio rastrear todos sus experimentos y barridos de hiperparámetros utilizando Weights & Biases (W&B). Todo el código debe adherirse a los principios SOLID y buenas prácticas de ingeniería de software (TDD, uso de linters, modularidad).

---

### Herramienta Incluida: Explorador del Conjunto de Datos

Para facilitar el arranque del proyecto, este repositorio incluye un script auxiliar llamado `dataset_viewer.py`. Esta herramienta se encarga de verificar la existencia del conjunto de datos BCCD, descargarlo si es necesario, y proveer una interfaz gráfica ligera para explorar las imágenes y sus anotaciones.

#### Requisitos previos
Asegúrense de tener instaladas las dependencias gráficas básicas:
```bash
pip install matplotlib pillow
```
