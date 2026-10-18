# Actividad 03 — Algoritmos de Enjambre en Machine Learning

Este repositorio contiene la implementación, hecha desde cero (sin librerías de swarm intelligence ya construidas), de dos algoritmos de enjambre —**ABC** (Artificial Bee Colony) y **PSO** (Particle Swarm Optimization)— aplicados a cuatro etapas distintas del ciclo de vida de un modelo de aprendizaje automático.

**Autora:** Maria Vanesa Yucra Mendoza
**Curso:** Algoritmos de Enjambre — EPIS, Universidad Nacional del Altiplano
**Trabajo:** Individual

## ¿Qué problema resuelve cada script?

### 1. `feature_selection_abc.py` — Feature selection con ABC

Un dataset puede tener variables (features) redundantes o poco útiles para un modelo de clasificación. Este script usa el algoritmo **ABC (Colonia Artificial de Abejas)** para encontrar, de forma automática, el subconjunto de características que mejor clasifica el dataset **Breast Cancer Wisconsin** (569 muestras, diagnóstico maligno/benigno).

Cada "fuente de alimento" que exploran las abejas representa un posible subconjunto de las 30 características disponibles. El algoritmo evalúa cada subconjunto entrenando un modelo (regresión logística) y midiendo su exactitud mediante validación cruzada, premiando además a los subconjuntos más pequeños (más simples). Las abejas empleadas, observadoras y exploradoras —los tres roles clásicos de ABC— van refinando y descartando subconjuntos iteración tras iteración.

**Resultado:** redujo las 30 características a solo 9, manteniendo 95.61% de exactitud (vs. 98.25% usando las 30 originales).

### 2. `pso_hyperparameter_tuning.py` — Hyperparameter tuning con PSO

Un modelo SVM (Support Vector Machine) depende de dos hiperparámetros clave —`C` y `gamma`— que normalmente se ajustan a prueba y error o con una búsqueda exhaustiva (Grid Search). Este script usa **PSO (Optimización por Enjambre de Partículas)** para encontrar automáticamente la mejor combinación de ambos, sobre el mismo dataset Breast Cancer Wisconsin.

Cada partícula del enjambre "vuela" por el espacio de posibles valores de `C` y `gamma` (en escala logarítmica, porque estos parámetros funcionan mejor en potencias de 10). Cada partícula se atrae hacia su propio mejor resultado histórico y hacia el mejor resultado de todo el enjambre, hasta converger en la combinación óptima.

**Resultado:** encontró C = 596.67 y gamma = 0.00010, logrando 98.25% de exactitud en el conjunto de prueba.

### 3. `nn_training_pso.py` — Red neuronal entrenada sin backpropagation

Normalmente una red neuronal se entrena calculando gradientes y aplicando backpropagation. Este script demuestra que es posible entrenar una red completa **sin ningún gradiente**, usando PSO como optimizador directo de los pesos y bias.

Se construye una red de 3 capas (30 entradas – 6 neuronas ocultas – 1 salida). Aquí la idea central es que **cada partícula del enjambre ES una red neuronal completa**: su posición (un vector de 193 números) contiene todos los pesos y bias de esa red. La función de aptitud simplemente hace una propagación hacia adelante (forward pass) con esos pesos y mide qué tan bien clasifica. El enjambre completo —40 redes distintas compitiendo entre sí— converge hacia la mejor combinación de pesos tras 150 iteraciones.

**Resultado:** 100% de exactitud en entrenamiento y 94.74% en el conjunto de prueba, confirmando que PSO logra entrenar la red exitosamente sin backpropagation.

### 4. `clustering_pso.py` — Clustering con PSO (parte bono)

Como alternativa adicional, este script aplica PSO a un problema de **aprendizaje no supervisado**: agrupar datos del dataset **Iris** (3 especies de flor) en 3 clusters, sin usar las etiquetas reales durante la búsqueda.

Cada partícula representa la ubicación de los 3 centroides. La función de aptitud mide el "error de cuantización": qué tan lejos está, en promedio, cada punto de su centroide asignado. El enjambre mueve los centroides hasta minimizar ese error, de forma similar a K-Means pero usando PSO en vez del algoritmo de Lloyd.

**Resultado:** error de cuantización final de 0.7466 y un Adjusted Rand Index de 0.45 frente a las especies reales (coincidencia parcial, esperable porque dos de las tres especies de Iris se superponen en el espacio de características).

## Requisitos e instalación

```bash
pip install numpy scikit-learn
```

Probado con Python 3.10+, numpy y scikit-learn (cualquier versión reciente).

## Cómo ejecutar

Cada script es independiente y se ejecuta directamente desde la terminal; al correrlo se imprime el progreso de cada iteración y los resultados finales:

```bash
python feature_selection_abc.py
python pso_hyperparameter_tuning.py
python nn_training_pso.py
python clustering_pso.py
```

También se pueden copiar y pegar en una celda de Google Colab o Jupyter Notebook sin modificaciones.

## Ciclo del algoritmo de enjambre seguido en los 4 scripts

Los cuatro scripts siguen la misma estructura conceptual, pedida como parte de la evaluación:

1. **Representación** de la partícula / fuente de alimento — cómo se codifica una solución candidata como un vector numérico.
2. **Inicialización** del enjambre — cómo se generan las soluciones candidatas iniciales.
3. **Función de aptitud** (fitness) — cómo se mide qué tan buena es una solución.
4. **Comportamiento** de cada agente — cómo se mueve/actualiza cada partícula o abeja en cada iteración.
5. **Evolución** iterativa del enjambre — cómo el enjambre completo mejora iteración tras iteración.
6. **Finalización** — el criterio de parada y cómo se extrae la mejor solución encontrada.

Cada una de estas 6 etapas está marcada explícitamente con comentarios numerados dentro del código de los 4 scripts, y explicada con mayor detalle matemático en `Informe_Actividad03_Enjambre.docx`.

## Resumen de resultados

| Parte | Algoritmo | Dataset | Resultado principal |
|---|---|---|---|
| Feature selection | ABC | Breast Cancer Wisconsin | 9 de 30 características — 95.61% accuracy |
| Hyperparameter tuning | PSO | Breast Cancer Wisconsin | C = 596.67, gamma = 0.00010 — 98.25% accuracy |
| Entrenamiento de red neuronal | PSO | Breast Cancer Wisconsin | Red 30‑6‑1 — 94.74% accuracy (sin backpropagation) |
| Clustering (bono) | PSO | Iris | 3 clusters — Adjusted Rand Index = 0.45 |
