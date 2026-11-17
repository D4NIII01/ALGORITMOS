# Nivel 3: Monte Carlo en GPU con numba.cuda
# Requiere haber corrido antes nivel2.py (usa Evaluacion3/nivel2/tiempos_cpu.csv)
import math
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # solo guarda imagenes, no abre ventanas
import matplotlib.pyplot as plt
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

base = Path(__file__).parent / "Evaluacion3"
carpeta = base / "nivel3"                       # Evaluacion3/nivel3
carpeta.mkdir(parents=True, exist_ok=True)

HILOS_POR_BLOQUE = 256
MAX_HILOS = 100_000          # maximo de hilos que usamos


@cuda.jit
def kernel(rng_states, muestras_por_hilo, parciales):
    i = cuda.grid(1)                              # indice global del hilo
    if i < parciales.size:
        suma = 0.0
        for k in range(muestras_por_hilo):
            # numero uniforme en [0,1) de SU estado -> lo llevamos a [-1,1)
            x = 2.0 * xoroshiro128p_uniform_float64(rng_states, i) - 1.0
            y = 2.0 * xoroshiro128p_uniform_float64(rng_states, i) - 1.0
            r2 = x * x + y * y
            if r2 <= 1.0:
                r = math.sqrt(r2)
                suma += (1.0 + r) * r2
        parciales[i] = suma                       # suma parcial de este hilo


def mc_gpu(N, rng_states, n_hilos, d_parciales):
    muestras_por_hilo = N // n_hilos
    bloques = (n_hilos + HILOS_POR_BLOQUE - 1) // HILOS_POR_BLOQUE
    kernel[bloques, HILOS_POR_BLOQUE](rng_states, muestras_por_hilo, d_parciales)
    cuda.synchronize()
    parciales = d_parciales.copy_to_host()
    return 4.0 * parciales.sum() / N              # reduccion final en CPU


def medir_gpu(N, repeticiones=3):
    n_hilos = min(N, MAX_HILOS)
    rng_states = create_xoroshiro128p_states(n_hilos, seed=12345)   # un estado por hilo
    d_parciales = cuda.device_array(n_hilos, dtype=np.float64)
    mc_gpu(N, rng_states, n_hilos, d_parciales)                     # calentamiento (compila)
    mejor = 1e99
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        I = mc_gpu(N, rng_states, n_hilos, d_parciales)
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor, I


def sigma_I(N):
    """Error estadistico de I para N puntos (estimado con una muestra en CPU)."""
    x = np.random.uniform(-1, 1, 10**6)
    y = np.random.uniform(-1, 1, 10**6)
    r = np.sqrt(x**2 + y**2)
    f = (1 + r) * r**2 * (r <= 1)
    return 4.0 * np.std(f) / math.sqrt(N)


I_exacto = 0.9 * np.pi
cpu = np.loadtxt(base / "nivel2" / "tiempos_cpu.csv", delimiter=",", skiprows=1)

print(f"{'N':>10} {'T gpu (s)':>11} {'I gpu':>10} {'|I-Iexacto|':>12} {'3*sigma':>10} {'OK?':>5} {'S gpu':>9}")
filas = []
speedup_gpu = []
for N, t_sec, t_par in cpu:
    N = int(N)
    t_gpu, I_gpu = medir_gpu(N)
    err = abs(I_gpu - I_exacto)
    tres_sigma = 3 * sigma_I(N)
    ok = "si" if err < tres_sigma else "no"
    S = t_sec / t_gpu
    speedup_gpu.append(S)
    filas.append([N, t_sec, t_par, t_gpu, t_sec / t_par, S, I_gpu, err, tres_sigma])
    print(f"{N:>10} {t_gpu:>11.5f} {I_gpu:>10.5f} {err:>12.2e} {tres_sigma:>10.2e} {ok:>5} {S:>9.2f}")

np.savetxt(carpeta / "resultados_nivel3.csv", filas, delimiter=",",
           header="N,t_sec,t_par,t_gpu,speedup_cpu,speedup_gpu,I_gpu,error_abs,tres_sigma",
           comments="")

# Grafica de speedup: CPU paralela y GPU
plt.semilogx(cpu[:, 0], cpu[:, 1] / cpu[:, 2], "o-", label="CPU paralela")
plt.semilogx(cpu[:, 0], speedup_gpu, "s-", label="GPU (CUDA)")
plt.axhline(1.0, color="gray", linestyle="--")
plt.xlabel("N")
plt.ylabel("Speedup respecto a CPU secuencial")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(carpeta / "speedup_total.png", dpi=150)

print(f"\nResultados guardados en: {carpeta}")
