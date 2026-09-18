"""
Actividad 03 - Algoritmos de enjambre en Machine Learning
Parte c) Entrenamiento de una red neuronal SIN backpropagation, usando PSO


"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


# ---------------------------------------------------------------------------
# 0. DATOS
# ---------------------------------------------------------------------------
def cargar_datos():
    data = load_breast_cancer()
    X, y = data.data, data.target
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# ARQUITECTURA DE LA RED (3 capas: entrada -> oculta -> salida)
# ---------------------------------------------------------------------------
N_ENTRADA = 30
N_OCULTA = 6
N_SALIDA = 1

# Cantidad total de numeros (pesos + bias) que describen la red completa
N_PESOS_TOTAL = (N_ENTRADA * N_OCULTA) + N_OCULTA + (N_OCULTA * N_SALIDA) + N_SALIDA


# ---------------------------------------------------------------------------
# 1. REPRESENTACION DE LA PARTICULA
# ---------------------------------------------------------------------------
# Cada particula es un vector de longitud N_PESOS_TOTAL que contiene, en
# orden, todos los pesos y bias de la red:
#   [ W1 (30x6 aplanado) | b1 (6) | W2 (6x1 aplanado) | b2 (1) ]
def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def decodificar_pesos(particula):
    """Reconstruye las matrices W1, b1, W2, b2 a partir del vector plano."""
    idx = 0
    W1 = particula[idx: idx + N_ENTRADA * N_OCULTA].reshape(N_ENTRADA, N_OCULTA)
    idx += N_ENTRADA * N_OCULTA
    b1 = particula[idx: idx + N_OCULTA]
    idx += N_OCULTA
    W2 = particula[idx: idx + N_OCULTA * N_SALIDA].reshape(N_OCULTA, N_SALIDA)
    idx += N_OCULTA * N_SALIDA
    b2 = particula[idx: idx + N_SALIDA]
    return W1, b1, W2, b2


def forward_pass(X, particula):
    """
    Propagacion hacia adelante de la red (entrada -> oculta -> salida).
    Esta es la UNICA operacion que se hace con la red: nunca se retropropaga
    ningun error ni se calcula ningun gradiente.
    """
    W1, b1, W2, b2 = decodificar_pesos(particula)
    capa_oculta = np.tanh(X @ W1 + b1)          # activacion tanh en la capa oculta
    salida = sigmoide(capa_oculta @ W2 + b2)     # activacion sigmoide en la salida
    return salida


# ---------------------------------------------------------------------------
# 2. FUNCION DE APTITUD (FITNESS)
# ---------------------------------------------------------------------------
def fitness(particula, X_train, y_train):
    """
    Mide que tan buena es una particula (= una red neuronal completa)
    calculando la entropia cruzada binaria entre sus predicciones y las
    etiquetas reales. PSO MINIMIZA este valor: mientras mas bajo, la red
    clasifica mejor.
    """
    predicciones = forward_pass(X_train, particula)
    epsilon = 1e-9
    predicciones = np.clip(predicciones, epsilon, 1 - epsilon)
    perdida = -np.mean(
        y_train * np.log(predicciones) + (1 - y_train) * np.log(1 - predicciones)
    )
    return perdida


def accuracy(X, y, particula):
    predicciones = forward_pass(X, particula) > 0.5
    return (predicciones == y).mean()


# ---------------------------------------------------------------------------
# 3. INICIALIZACION DEL ENJAMBRE
# ---------------------------------------------------------------------------
def inicializar_enjambre(n_particulas):
    # Pesos iniciales pequeños (tipico en redes neuronales, evita saturar tanh/sigmoide)
    posiciones = np.random.uniform(-1, 1, size=(n_particulas, N_PESOS_TOTAL))
    velocidades = np.random.uniform(-0.1, 0.1, size=(n_particulas, N_PESOS_TOTAL))
    return posiciones, velocidades


# ---------------------------------------------------------------------------
# 4. COMPORTAMIENTO DE LA PARTICULA
# ---------------------------------------------------------------------------
def actualizar_particula(posicion, velocidad, pbest, gbest, w, c1, c2):
    r1 = np.random.rand(N_PESOS_TOTAL)
    r2 = np.random.rand(N_PESOS_TOTAL)
    nueva_velocidad = (
        w * velocidad
        + c1 * r1 * (pbest - posicion)
        + c2 * r2 * (gbest - posicion)
    )
    nueva_velocidad = np.clip(nueva_velocidad, -1, 1)  # evita explosion de velocidad
    nueva_posicion = posicion + nueva_velocidad
    nueva_posicion = np.clip(nueva_posicion, -5, 5)     # limites razonables para pesos
    return nueva_posicion, nueva_velocidad


# ---------------------------------------------------------------------------
# 5. EVOLUCION DEL ENJAMBRE
# ---------------------------------------------------------------------------
def pso_entrenar_red(X_train, y_train, n_particulas=40, n_iteraciones=150, w=0.6, c1=1.5, c2=1.5):
    posiciones, velocidades = inicializar_enjambre(n_particulas)

    pbest_posiciones = posiciones.copy()
    pbest_valores = np.array([fitness(p, X_train, y_train) for p in posiciones])

    indice_gbest = np.argmin(pbest_valores)
    gbest_posicion = pbest_posiciones[indice_gbest].copy()
    gbest_valor = pbest_valores[indice_gbest]

    for iteracion in range(n_iteraciones):
        for i in range(n_particulas):
            posiciones[i], velocidades[i] = actualizar_particula(
                posiciones[i], velocidades[i], pbest_posiciones[i], gbest_posicion, w, c1, c2
            )
            valor_actual = fitness(posiciones[i], X_train, y_train)

            if valor_actual < pbest_valores[i]:
                pbest_valores[i] = valor_actual
                pbest_posiciones[i] = posiciones[i].copy()
                if valor_actual < gbest_valor:
                    gbest_valor = valor_actual
                    gbest_posicion = posiciones[i].copy()

        if (iteracion + 1) % 10 == 0 or iteracion == 0:
            acc_train = accuracy(X_train, y_train, gbest_posicion)
            print(
                f"Iteracion {iteracion + 1:03d}/{n_iteraciones} | "
                f"perdida = {gbest_valor:.4f} | accuracy train = {acc_train:.4f}"
            )

    # -----------------------------------------------------------------
    # 6. FINALIZACION
    # -----------------------------------------------------------------
    # Criterio de parada: numero fijo de iteraciones (150). Al finalizar,
    # gbest_posicion contiene los pesos y bias definitivos de la red.
    return gbest_posicion, gbest_valor


# ---------------------------------------------------------------------------
# EJECUCION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = cargar_datos()

    print(f"Entrenando red neuronal ({N_ENTRADA}-{N_OCULTA}-{N_SALIDA}) con PSO ")
    print(f"({N_PESOS_TOTAL} pesos y bias en total, sin backpropagation)\n")

    mejores_pesos, perdida_final = pso_entrenar_red(X_train, y_train)

    acc_train = accuracy(X_train, y_train, mejores_pesos)
    acc_test = accuracy(X_test, y_test, mejores_pesos)

    print("\n=== Resultado final ===")
    print(f"Perdida (binary cross-entropy) final: {perdida_final:.4f}")
    print(f"Accuracy en entrenamiento: {acc_train:.4f}")
    print(f"Accuracy en prueba (holdout): {acc_test:.4f}")
