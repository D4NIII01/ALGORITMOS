# Nivel 2: Monte Carlo secuencial vs paralelo en CPU con Numba
import os
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # solo guarda imagenes, no abre ventanas
import matplotlib.pyplot as plt
from numba import njit, prange

# Carpeta de resultados: Evaluacion3/nivel2 (junto a este script)
carpeta = Path(__file__).parent / "Evaluacion3" / "nivel2"
carpeta.mkdir(parents=True, exist_ok=True)


@njit
def mc_secuencial(N):
    suma = 0.0
    for i in range(N):
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        r2 = x * x + y * y
        if r2 <= 1.0:                       # funcion indicadora
            r = np.sqrt(r2)
            suma += (1.0 + r) * r2          # rho * r^2
    return 4.0 * suma / N                   # A = 4


@njit(parallel=True)
def mc_paralelo(N):
    suma = 0.0
    for i in prange(N):                     # las iteraciones se reparten entre los nucleos
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        r2 = x * x + y * y
        if r2 <= 1.0:
            r = np.sqrt(r2)
            suma += (1.0 + r) * r2          # Numba suma las parciales de cada hilo
    return 4.0 * suma / N


def medir(funcion, N, repeticiones=3):
    """Devuelve (mejor tiempo, resultado)."""
    mejor = 1e99
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        resultado = funcion(N)
        t1 = time.perf_counter()
        mejor = min(mejor, t1 - t0)
    return mejor, resultado


I_exacto = 0.9 * np.pi

# Calentamiento: la primera llamada compila (JIT); no se mide
mc_secuencial(1000)
mc_paralelo(1000)

Ns = [10**k for k in range(4, 9)]           # 10^4 ... 10^8
nucleos = os.cpu_count()
print(f"Nucleos de la maquina: {nucleos}\n")
print(f"{'N':>10} {'T sec (s)':>12} {'T par (s)':>12} {'Speedup':>9} {'I sec':>10} {'I par':>10}")

filas = []
for N in Ns:
    t_sec, I_sec = medir(mc_secuencial, N)
    t_par, I_par = medir(mc_paralelo, N)
    S = t_sec / t_par
    filas.append([N, t_sec, t_par, S, I_sec, I_par])
    print(f"{N:>10} {t_sec:>12.5f} {t_par:>12.5f} {S:>9.2f} {I_sec:>10.5f} {I_par:>10.5f}")

print(f"\nI exacto = {I_exacto:.5f}")

# Tabla completa (para el informe) y tiempos que lee el nivel 3
np.savetxt(carpeta / "resultados_nivel2.csv", filas, delimiter=",",
           header="N,t_sec,t_par,speedup,I_sec,I_par", comments="")
np.savetxt(carpeta / "tiempos_cpu.csv", [f[:3] for f in filas], delimiter=",",
           header="N,t_sec,t_par", comments="")
with open(carpeta / "nucleos.txt", "w") as f:
    f.write(f"Nucleos de la maquina: {nucleos}\n")

filas = np.array(filas)
plt.semilogx(filas[:, 0], filas[:, 3], "o-", label="CPU paralela (Numba)")
plt.axhline(1.0, color="gray", linestyle="--", label="speedup = 1")
plt.xlabel("N")
plt.ylabel("Speedup  S = T_sec / T_par")
plt.title(f"Speedup en CPU ({nucleos} nucleos)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(carpeta / "speedup_cpu.png", dpi=150)

print(f"\nResultados guardados en: {carpeta}")
