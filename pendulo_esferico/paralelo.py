import argparse
import os
import time
from multiprocessing import Pool, cpu_count

from modelo import construir_modelo, integrar, condiciones_iniciales, guardar

_MODELO = None


def _init_worker():
    global _MODELO
    _MODELO = construir_modelo()


def _tarea(args):
    S0, tfinal, npuntos = args
    return integrar(S0, _MODELO, tfinal, npuntos)


def ejecutar_paralelo(n, procesos, tfinal=40.0, npuntos=1001, semilla=0):
    S0s = condiciones_iniciales(n, semilla)
    tareas = [(S0, tfinal, npuntos) for S0 in S0s]
    with Pool(procesos, initializer=_init_worker) as pool:
        return pool.map(_tarea, tareas, chunksize=1)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--procesos', type=int, default=None)
    p.add_argument('--tfinal', type=float, default=40.0)
    p.add_argument('--npuntos', type=int, default=1001)
    p.add_argument('--semilla', type=int, default=0)
    p.add_argument('--salida', default=os.path.join(base, 'data'))
    a = p.parse_args()

    procesos = a.procesos or min(cpu_count(), a.n)
    print(f"Simulando {a.n} trayectorias en {procesos} procesos")


    t0 = time.perf_counter()
    resultados = ejecutar_paralelo(a.n, procesos, a.tfinal, a.npuntos, a.semilla)
    t = time.perf_counter() - t0

    for k, (_, deriva, ok) in enumerate(resultados):
        print(f"Trayectoria {k}: {'OK' if ok else 'FALLO'}, deriva de energia = {deriva:.2e}")
    guardadas = guardar(resultados, a.salida)
    print(f"Guardadas {guardadas} trayectorias en {a.salida}")
    print(f"Tiempo total ({procesos} procesos): {t:.2f} s")


if __name__ == '__main__':
    main()
