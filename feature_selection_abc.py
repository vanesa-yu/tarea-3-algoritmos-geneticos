"""
Actividad 03 - Algoritmos de enjambre en Machine Learning
Parte a) Feature Selection usando ABC (Artificial Bee Colony)


"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


# ---------------------------------------------------------------------------
# 0. DATOS
# ---------------------------------------------------------------------------
def cargar_datos():
    data = load_breast_cancer()
    X, y = data.data, data.target
    nombres_features = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, nombres_features


# ---------------------------------------------------------------------------
# 1. REPRESENTACION DE LA FUENTE DE ALIMENTO (particula / solucion)
# ---------------------------------------------------------------------------
# En ABC cada "fuente de alimento" (food source) representa una solucion
# candidata. Aqui cada fuente es un vector binario de longitud = numero de
# features (30), donde 1 = la caracteristica se incluye, 0 = se descarta.
#
# Para permitir que ABC (que naturalmente trabaja con numeros continuos)
# opere sobre un problema binario, cada fuente se guarda como un vector
# continuo en [0, 1] y se binariza con un umbral de 0.5 al momento de evaluar.
N_FEATURES = 30
LIMITE_INFERIOR = 0.0
LIMITE_SUPERIOR = 1.0


def binarizar(vector_continuo):
    """Convierte el vector continuo [0,1]^30 en un vector binario de seleccion."""
    seleccion = (vector_continuo > 0.5).astype(int)
    if seleccion.sum() == 0:
        # Evita el caso degenerado de no seleccionar ninguna caracteristica
        seleccion[np.argmax(vector_continuo)] = 1
    return seleccion


# ---------------------------------------------------------------------------
# 2. FUNCION DE APTITUD (FITNESS)
# ---------------------------------------------------------------------------
def fitness(vector_continuo, X_train, y_train, alpha=0.9):
    """
    Evalua una fuente de alimento (subconjunto de features).


    """
    seleccion = binarizar(vector_continuo)
    indices = np.where(seleccion == 1)[0]

    X_subconjunto = X_train[:, indices]
    modelo = LogisticRegression(max_iter=2000)

    try:
        accuracy = cross_val_score(modelo, X_subconjunto, y_train, cv=5, n_jobs=-1).mean()
    except Exception:
        accuracy = 0.0

    proporcion_usada = len(indices) / N_FEATURES
    return alpha * accuracy + (1 - alpha) * (1 - proporcion_usada)


# ---------------------------------------------------------------------------
# 3. INICIALIZACION DEL ENJAMBRE (fuentes de alimento)
# ---------------------------------------------------------------------------
def inicializar_fuentes(n_fuentes):
    fuentes = np.random.uniform(LIMITE_INFERIOR, LIMITE_SUPERIOR, size=(n_fuentes, N_FEATURES))
    contador_intentos = np.zeros(n_fuentes)  # "trial counter" de cada fuente
    return fuentes, contador_intentos


# ---------------------------------------------------------------------------
# 4. COMPORTAMIENTO DE LAS ABEJAS
# ---------------------------------------------------------------------------
def generar_solucion_vecina(fuentes, i):
    
    
    n_fuentes = fuentes.shape[0]
    j = np.random.choice([idx for idx in range(n_fuentes) if idx != i])
    dimension = np.random.randint(N_FEATURES)

    phi = np.random.uniform(-1, 1)
    nueva_fuente = fuentes[i].copy()
    nueva_fuente[dimension] = fuentes[i, dimension] + phi * (fuentes[i, dimension] - fuentes[j, dimension])
    nueva_fuente = np.clip(nueva_fuente, LIMITE_INFERIOR, LIMITE_SUPERIOR)
    return nueva_fuente


def fase_abejas_empleadas(fuentes, contador_intentos, valores_fitness, X_train, y_train):
    """Cada abeja empleada explora alrededor de SU fuente de alimento asignada."""
    n_fuentes = fuentes.shape[0]
    for i in range(n_fuentes):
        candidata = generar_solucion_vecina(fuentes, i)
        fitness_candidata = fitness(candidata, X_train, y_train)

        if fitness_candidata > valores_fitness[i]:
            fuentes[i] = candidata
            valores_fitness[i] = fitness_candidata
            contador_intentos[i] = 0  # la fuente mejoro, se reinicia su contador
        else:
            contador_intentos[i] += 1  # la fuente no mejoro, incrementa su contador

    return fuentes, contador_intentos, valores_fitness


def fase_abejas_observadoras(fuentes, contador_intentos, valores_fitness, X_train, y_train):
    
    n_fuentes = fuentes.shape[0]
    fitness_positivo = valores_fitness - valores_fitness.min() + 1e-6
    probabilidades = fitness_positivo / fitness_positivo.sum()

    for _ in range(n_fuentes):
        i = np.random.choice(n_fuentes, p=probabilidades)
        candidata = generar_solucion_vecina(fuentes, i)
        fitness_candidata = fitness(candidata, X_train, y_train)

        if fitness_candidata > valores_fitness[i]:
            fuentes[i] = candidata
            valores_fitness[i] = fitness_candidata
            contador_intentos[i] = 0
        else:
            contador_intentos[i] += 1

    return fuentes, contador_intentos, valores_fitness


def fase_abejas_exploradoras(fuentes, contador_intentos, valores_fitness, X_train, y_train, limite_abandono):
    """
    Si una fuente de alimento no mejora tras 'limite_abandono' intentos, se
    considera agotada: la abeja empleada asignada se convierte en
    exploradora y la fuente se reemplaza por una nueva posicion aleatoria.
    """
    n_fuentes = fuentes.shape[0]
    for i in range(n_fuentes):
        if contador_intentos[i] > limite_abandono:
            fuentes[i] = np.random.uniform(LIMITE_INFERIOR, LIMITE_SUPERIOR, size=N_FEATURES)
            valores_fitness[i] = fitness(fuentes[i], X_train, y_train)
            contador_intentos[i] = 0

    return fuentes, contador_intentos, valores_fitness


# ---------------------------------------------------------------------------
# 5. EVOLUCION DEL ENJAMBRE (ciclo principal de ABC)
# ---------------------------------------------------------------------------
def abc_feature_selection(X_train, y_train, n_fuentes=15, n_iteraciones=25, limite_abandono=8):
    fuentes, contador_intentos = inicializar_fuentes(n_fuentes)
    valores_fitness = np.array([fitness(f, X_train, y_train) for f in fuentes])

    indice_mejor = np.argmax(valores_fitness)
    mejor_fuente_global = fuentes[indice_mejor].copy()
    mejor_fitness_global = valores_fitness[indice_mejor]

    for iteracion in range(n_iteraciones):
        fuentes, contador_intentos, valores_fitness = fase_abejas_empleadas(
            fuentes, contador_intentos, valores_fitness, X_train, y_train
        )
        fuentes, contador_intentos, valores_fitness = fase_abejas_observadoras(
            fuentes, contador_intentos, valores_fitness, X_train, y_train
        )
        fuentes, contador_intentos, valores_fitness = fase_abejas_exploradoras(
            fuentes, contador_intentos, valores_fitness, X_train, y_train, limite_abandono
        )

        indice_mejor = np.argmax(valores_fitness)
        if valores_fitness[indice_mejor] > mejor_fitness_global:
            mejor_fitness_global = valores_fitness[indice_mejor]
            mejor_fuente_global = fuentes[indice_mejor].copy()

        seleccion_actual = binarizar(mejor_fuente_global)
        print(
            f"Iteracion {iteracion + 1:02d}/{n_iteraciones} | "
            f"mejor fitness = {mejor_fitness_global:.4f} | "
            f"features seleccionadas = {seleccion_actual.sum()}/{N_FEATURES}"
        )

    # -----------------------------------------------------------------
    # 6. FINALIZACION
    # -----------------------------------------------------------------
    # Criterio de parada: numero fijo de iteraciones (25).
    seleccion_final = binarizar(mejor_fuente_global)
    return seleccion_final, mejor_fitness_global


# ---------------------------------------------------------------------------
# EJECUCION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, nombres_features = cargar_datos()

    print("Buscando el mejor subconjunto de caracteristicas con ABC...\n")
    seleccion, mejor_fitness = abc_feature_selection(X_train, y_train)

    indices_seleccionados = np.where(seleccion == 1)[0]
    print(f"\n=== Mejor subconjunto encontrado ({len(indices_seleccionados)} de {N_FEATURES} features) ===")
    for idx in indices_seleccionados:
        print(f"  - {nombres_features[idx]}")

    # Entrenamos el modelo final solo con las features seleccionadas
    modelo_final = LogisticRegression(max_iter=2000)
    modelo_final.fit(X_train[:, indices_seleccionados], y_train)
    accuracy_test = modelo_final.score(X_test[:, indices_seleccionados], y_test)
    accuracy_todas = LogisticRegression(max_iter=2000).fit(X_train, y_train).score(X_test, y_test)

    print(f"\nAccuracy en test usando {len(indices_seleccionados)} features seleccionadas: {accuracy_test:.4f}")
    print(f"Accuracy en test usando las 30 features (referencia): {accuracy_todas:.4f}")
