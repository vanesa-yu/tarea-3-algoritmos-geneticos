"""
Actividad 03 - Algoritmos de enjambre en Machine Learning
Parte b) Hyperparameter tuning usando PSO (Particle Swarm Optimization)


"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Semilla fija para que los resultados sean reproducibles
np.random.seed(42)


# ---------------------------------------------------------------------------
# 0. DATOS
# ---------------------------------------------------------------------------
def cargar_datos():
    """Carga el dataset y separa un conjunto de prueba (holdout) final."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Estandarizamos: SVM es sensible a la escala de las variables
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 1. REPRESENTACION DE LA PARTICULA
# ---------------------------------------------------------------------------
# Cada particula es un vector real de 2 dimensiones:
#   posicion = [log10(C), log10(gamma)]
#
# Usamos escala logaritmica porque C y gamma en SVM funcionan mejor
# explorados en potencias de 10 (ej. 0.001, 0.01, 0.1, 1, 10, 100)
# en vez de una escala lineal.
DIMENSIONES = 2
LIMITE_INFERIOR = np.array([-3, -4])   # log10(C) en [-3, 3] -> C en [0.001, 1000]
LIMITE_SUPERIOR = np.array([3, 1])     # log10(gamma) en [-4, 1] -> gamma en [0.0001, 10]


def decodificar_particula(posicion):
    """Convierte la posicion (espacio log) en hiperparametros reales C y gamma."""
    C = 10 ** posicion[0]
    gamma = 10 ** posicion[1]
    return C, gamma


# ---------------------------------------------------------------------------
# 2. FUNCION DE APTITUD (FITNESS)
# ---------------------------------------------------------------------------
def fitness(posicion, X_train, y_train):
    """
    Evalua que tan buena es una combinacion de hiperparametros.

    """
    C, gamma = decodificar_particula(posicion)
    modelo = SVC(C=C, gamma=gamma, kernel="rbf")

    try:
        scores = cross_val_score(modelo, X_train, y_train, cv=5, n_jobs=-1)
        accuracy_promedio = scores.mean()
    except Exception:
        # Si una combinacion de hiperparametros produce un error numerico
        accuracy_promedio = 0.0

    return 1.0 - accuracy_promedio


# ---------------------------------------------------------------------------
# 3. INICIALIZACION DEL ENJAMBRE
# ---------------------------------------------------------------------------
def inicializar_enjambre(n_particulas):
    """
    Crea el enjambre inicial:
      - posiciones aleatorias uniformes dentro de los limites de busqueda
      - velocidades iniciales pequeñas (aleatorias)
      - pbest (mejor posicion personal) = posicion inicial de cada particula
    """
    posiciones = np.random.uniform(
        LIMITE_INFERIOR, LIMITE_SUPERIOR, size=(n_particulas, DIMENSIONES)
    )
    rango = LIMITE_SUPERIOR - LIMITE_INFERIOR
    velocidades = np.random.uniform(-1, 1, size=(n_particulas, DIMENSIONES)) * rango * 0.1

    return posiciones, velocidades


# ---------------------------------------------------------------------------
# 4. COMPORTAMIENTO DE LA PARTICULA (actualizacion de velocidad y posicion)
# ---------------------------------------------------------------------------
def actualizar_particula(posicion, velocidad, pbest, gbest, w, c1, c2):
   
    r1 = np.random.rand(DIMENSIONES)
    r2 = np.random.rand(DIMENSIONES)

    nueva_velocidad = (
        w * velocidad
        + c1 * r1 * (pbest - posicion)
        + c2 * r2 * (gbest - posicion)
    )
    nueva_posicion = posicion + nueva_velocidad

    # Se mantiene la particula dentro de los limites validos de busqueda
    nueva_posicion = np.clip(nueva_posicion, LIMITE_INFERIOR, LIMITE_SUPERIOR)

    return nueva_posicion, nueva_velocidad


# ---------------------------------------------------------------------------
# 5. EVOLUCION DEL ENJAMBRE (ciclo principal de PSO)
# ---------------------------------------------------------------------------
def pso_hyperparameter_tuning(
    X_train, y_train,
    n_particulas=10,
    n_iteraciones=15,
    w=0.7, c1=1.5, c2=1.5,
):
    posiciones, velocidades = inicializar_enjambre(n_particulas)

    # pbest: mejor posicion encontrada por cada particula individualmente
    pbest_posiciones = posiciones.copy()
    pbest_valores = np.array([fitness(p, X_train, y_train) for p in posiciones])

    # gbest: mejor posicion encontrada por todo el enjambre
    indice_gbest = np.argmin(pbest_valores)
    gbest_posicion = pbest_posiciones[indice_gbest].copy()
    gbest_valor = pbest_valores[indice_gbest]

    historial_convergencia = [gbest_valor]

    for iteracion in range(n_iteraciones):
        for i in range(n_particulas):
            posiciones[i], velocidades[i] = actualizar_particula(
                posiciones[i], velocidades[i],
                pbest_posiciones[i], gbest_posicion,
                w, c1, c2,
            )

            valor_actual = fitness(posiciones[i], X_train, y_train)

            # Actualizar el mejor personal si mejora
            if valor_actual < pbest_valores[i]:
                pbest_valores[i] = valor_actual
                pbest_posiciones[i] = posiciones[i].copy()

                # Actualizar el mejor global si mejora
                if valor_actual < gbest_valor:
                    gbest_valor = valor_actual
                    gbest_posicion = posiciones[i].copy()

        historial_convergencia.append(gbest_valor)
        C, gamma = decodificar_particula(gbest_posicion)
        print(
            f"Iteracion {iteracion + 1:02d}/{n_iteraciones} | "
            f"mejor accuracy = {1 - gbest_valor:.4f} | "
            f"C = {C:.4f} | gamma = {gamma:.5f}"
        )

    # -----------------------------------------------------------------
    # 6. FINALIZACION
    # -----------------------------------------------------------------
    # Criterio de parada usado: numero fijo de iteraciones (n_iteraciones).
    # Alternativas posibles: detenerse si gbest no mejora en N iteraciones
    # seguidas (convergencia), o si se alcanza un accuracy objetivo.
    C_final, gamma_final = decodificar_particula(gbest_posicion)
    return {
        "C": C_final,
        "gamma": gamma_final,
        "mejor_accuracy_cv": 1 - gbest_valor,
        "historial_convergencia": historial_convergencia,
    }


# ---------------------------------------------------------------------------
# EJECUCION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = cargar_datos()

    print("Buscando los mejores hiperparametros de SVM con PSO...\n")
    resultado = pso_hyperparameter_tuning(X_train, y_train)

    print("\n=== Mejores hiperparametros encontrados ===")
    print(f"C     = {resultado['C']:.4f}")
    print(f"gamma = {resultado['gamma']:.5f}")
    print(f"Accuracy promedio (validacion cruzada) = {resultado['mejor_accuracy_cv']:.4f}")

    # Entrenamos el modelo final con TODO el set de entrenamiento
    # y evaluamos en el conjunto de prueba (holdout) nunca antes visto
    modelo_final = SVC(C=resultado["C"], gamma=resultado["gamma"], kernel="rbf")
    modelo_final.fit(X_train, y_train)
    accuracy_test = modelo_final.score(X_test, y_test)

    print(f"\nAccuracy en el conjunto de prueba (holdout): {accuracy_test:.4f}")
