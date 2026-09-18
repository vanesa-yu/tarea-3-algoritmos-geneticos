"""
Actividad 03 - Algoritmos de enjambre en Machine Learning
Parte d) (Alternativo / bono) Clustering usando PSO


"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

np.random.seed(42)

K = 3  # numero de clusters a formar


# ---------------------------------------------------------------------------
# 0. DATOS
# ---------------------------------------------------------------------------
def cargar_datos():
    data = load_iris()
    X = StandardScaler().fit_transform(data.data)
    y = data.target  # solo para verificar el resultado al final
    return X, y


# ---------------------------------------------------------------------------
# 1. REPRESENTACION DE LA PARTICULA
# ---------------------------------------------------------------------------
# Cada particula representa los K centroides completos, aplanados en un
# solo vector: [ centroide_1 (4 valores) | centroide_2 (4) | centroide_3 (4) ]
def decodificar_centroides(particula, n_features):
    return particula.reshape(K, n_features)


# ---------------------------------------------------------------------------
# 2. FUNCION DE APTITUD (FITNESS): error de cuantizacion
# ---------------------------------------------------------------------------
def asignar_clusters(X, centroides):
    """Asigna cada punto al centroide mas cercano (distancia euclidiana)."""
    distancias = np.linalg.norm(X[:, None, :] - centroides[None, :, :], axis=2)
    return np.argmin(distancias, axis=1), distancias


def fitness(particula, X):
    """
    El error de cuantizacion mide, en promedio, que tan lejos esta cada
    punto de su centroide asignado. PSO MINIMIZA este valor: mientras mas
    bajo, los clusters son mas compactos.
    """
    n_features = X.shape[1]
    centroides = decodificar_centroides(particula, n_features)
    asignaciones, distancias = asignar_clusters(X, centroides)

    error_total = 0.0
    for k in range(K):
        puntos_del_cluster = distancias[asignaciones == k, k]
        if len(puntos_del_cluster) > 0:
            error_total += puntos_del_cluster.mean()
        else:
            error_total += 10.0  # penaliza fuertemente un cluster vacio

    return error_total / K


# ---------------------------------------------------------------------------
# 3. INICIALIZACION DEL ENJAMBRE
# ---------------------------------------------------------------------------
def inicializar_enjambre(n_particulas, X):
    """
    Cada particula inicia con K centroides elegidos como puntos reales
    del dataset (tomados al azar, sin repetir dentro de la misma
    particula). Esto evita que los centroides iniciales caigan en zonas
    vacias del espacio y ayuda a que ningun cluster quede casi sin puntos.
    """
    n_muestras = X.shape[0]
    posiciones = np.array([
        X[np.random.choice(n_muestras, size=K, replace=False)].flatten()
        for _ in range(n_particulas)
    ])
    velocidades = np.random.uniform(-0.5, 0.5, size=(n_particulas, posiciones.shape[1]))
    return posiciones, velocidades


# ---------------------------------------------------------------------------
# 4. COMPORTAMIENTO DE LA PARTICULA
# ---------------------------------------------------------------------------
def actualizar_particula(posicion, velocidad, pbest, gbest, w, c1, c2):
    dim = posicion.shape[0]
    r1 = np.random.rand(dim)
    r2 = np.random.rand(dim)
    nueva_velocidad = (
        w * velocidad
        + c1 * r1 * (pbest - posicion)
        + c2 * r2 * (gbest - posicion)
    )
    nueva_velocidad = np.clip(nueva_velocidad, -1.0, 1.0)  # evita saltos demasiado grandes
    nueva_posicion = posicion + nueva_velocidad
    return nueva_posicion, nueva_velocidad


# ---------------------------------------------------------------------------
# 5. EVOLUCION DEL ENJAMBRE
# ---------------------------------------------------------------------------
def pso_clustering(X, n_particulas=30, n_iteraciones=100, w=0.72, c1=1.7, c2=1.7):
    posiciones, velocidades = inicializar_enjambre(n_particulas, X)

    pbest_posiciones = posiciones.copy()
    pbest_valores = np.array([fitness(p, X) for p in posiciones])

    indice_gbest = np.argmin(pbest_valores)
    gbest_posicion = pbest_posiciones[indice_gbest].copy()
    gbest_valor = pbest_valores[indice_gbest]

    for iteracion in range(n_iteraciones):
        for i in range(n_particulas):
            posiciones[i], velocidades[i] = actualizar_particula(
                posiciones[i], velocidades[i], pbest_posiciones[i], gbest_posicion, w, c1, c2
            )
            valor_actual = fitness(posiciones[i], X)

            if valor_actual < pbest_valores[i]:
                pbest_valores[i] = valor_actual
                pbest_posiciones[i] = posiciones[i].copy()
                if valor_actual < gbest_valor:
                    gbest_valor = valor_actual
                    gbest_posicion = posiciones[i].copy()

        if (iteracion + 1) % 5 == 0 or iteracion == 0:
            print(f"Iteracion {iteracion + 1:02d}/{n_iteraciones} | error de cuantizacion = {gbest_valor:.4f}")

    # -----------------------------------------------------------------
    # 6. FINALIZACION
    # -----------------------------------------------------------------
    # Criterio de parada: numero fijo de iteraciones (50).
    return gbest_posicion, gbest_valor


# ---------------------------------------------------------------------------
# EJECUCION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y_real = cargar_datos()
    n_features = X.shape[1]

    # Estrategia de reinicios multiples (multi-start): PSO puede converger a
    # un optimo local segun la inicializacion aleatoria del enjambre, asi
    # que se corren varias ejecuciones independientes y se conserva la mejor
    # (menor error de cuantizacion). Esto sigue siendo 100% PSO; solo se
    # repite el proceso para mitigar el efecto de una mala inicializacion.
    N_REINICIOS = 5
    mejor_error_global = np.inf
    mejores_centroides_planos = None

    for reinicio in range(N_REINICIOS):
        print(f"--- Reinicio {reinicio + 1}/{N_REINICIOS} ---")
        centroides_planos, error_final = pso_clustering(X)
        if error_final < mejor_error_global:
            mejor_error_global = error_final
            mejores_centroides_planos = centroides_planos

    error_final = mejor_error_global
    print(f"\nAgrupando datos en {K} clusters con PSO (mejor de {N_REINICIOS} reinicios)...")

    centroides = decodificar_centroides(mejores_centroides_planos, n_features)
    asignaciones, _ = asignar_clusters(X, centroides)

    print(f"\nError de cuantizacion final: {error_final:.4f}")
    print(f"Tamaño de cada cluster encontrado: {np.bincount(asignaciones)}")

    # Comparacion contra las especies reales (solo como verificacion, no se
    # uso esta informacion durante la busqueda de los clusters)
    ari = adjusted_rand_score(y_real, asignaciones)
    print(f"Adjusted Rand Index vs. especies reales: {ari:.4f} (1.0 = coincidencia perfecta)")
