# Nivel 1: Monte Carlo secuencial del momento de inercia de un disco
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # solo guarda imagenes, no abre ventanas
import matplotlib.pyplot as plt

# Carpeta de resultados: Evaluacion3/nivel1 (junto a este script)
carpeta = Path(__file__).parent / "Evaluacion3" / "nivel1"
carpeta.mkdir(parents=True, exist_ok=True)

rho0 = 1.0
R = 1.0
I_exacto = 0.9 * np.pi * rho0 * R**4


def monte_carlo(N):
    """Estima I con N puntos uniformes en el cuadrado [-R,R]x[-R,R]."""
    x = np.random.uniform(-R, R, N)
    y = np.random.uniform(-R, R, N)
    r = np.sqrt(x**2 + y**2)
    rho = rho0 * (1 + r / R)
    f = rho * r**2 * (r <= R)          # (r <= R) es la funcion indicadora
    A = (2 * R) ** 2                   # area del cuadrado
    I = A * np.mean(f)                 # estimador: area * promedio
    error_estadistico = A * np.std(f) / np.sqrt(N)
    return I, error_estadistico


Ns = [10**k for k in range(3, 8)]      # 10^3 ... 10^7
repeticiones = 20                      # promediamos el error para que la curva sea suave

print(f"I exacto = {I_exacto:.6f}\n")
print(f"{'N':>10} {'I (1 corrida)':>15} {'error abs.':>12} {'error estad.':>13} {'error medio(20)':>16}")

filas = []
errores_medios = []
for N in Ns:
    I, err_est = monte_carlo(N)
    err_abs = abs(I - I_exacto)
    errs = [abs(monte_carlo(N)[0] - I_exacto) for _ in range(repeticiones)]
    errores_medios.append(np.mean(errs))
    filas.append([N, I, err_abs, err_est, errores_medios[-1]])
    print(f"{N:>10} {I:>15.6f} {err_abs:>12.2e} {err_est:>13.2e} {errores_medios[-1]:>16.2e}")

# Pendiente de la recta ajustada en log-log
pendiente = np.polyfit(np.log10(Ns), np.log10(errores_medios), 1)[0]
print(f"\nPendiente ajustada en log-log: {pendiente:.2f} (esperada: -0.5)")

# Guardar tabla de resultados
np.savetxt(carpeta / "resultados_nivel1.csv", filas, delimiter=",",
           header="N,I,error_abs,error_estadistico,error_medio_20", comments="")

# Grafica log-log con recta de pendiente -1/2 que pasa por el primer punto
Ns_arr = np.array(Ns, dtype=float)
referencia = errores_medios[0] * (Ns_arr / Ns_arr[0]) ** (-0.5)

plt.loglog(Ns_arr, errores_medios, "o-", label="error medio de Monte Carlo")
plt.loglog(Ns_arr, referencia, "--", label="recta de pendiente -1/2")
plt.xlabel("N (numero de puntos)")
plt.ylabel("error absoluto")
plt.title("Convergencia del error")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig(carpeta / "convergencia.png", dpi=150)

print(f"\nResultados guardados en: {carpeta}")
