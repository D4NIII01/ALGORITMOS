import argparse
import os
import time

from modelo import construir_modelo, integrar, condiciones_iniciales, guardar


def ejecutar_serial(n, tfinal=40.0, npuntos=1001, semilla=0):
    modelo = construir_modelo()
    S0s = condiciones_iniciales(n, semilla)
    return [integrar(S0, modelo, tfinal, npuntos) for S0 in S0s]


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--tfinal', type=float, default=40.0)
    p.add_argument('--npuntos', type=int, default=1001)
    p.add_argument('--semilla', type=int, default=0)
    p.add_argument('--salida', default=os.path.join(base, 'data'))
    a = p.parse_args()

    t0 = time.perf_counter()
    resultados = ejecutar_serial(a.n, a.tfinal, a.npuntos, a.semilla)
    t = time.perf_counter() - t0

    for k, (_, deriva, ok) in enumerate(resultados):
        print(f"Trayectoria {k}: {'OK' if ok else 'FALLO'}, deriva de energia = {deriva:.2e}")
    guardadas = guardar(resultados, a.salida)
    print(f"Guardadas {guardadas} trayectorias en {a.salida}")
    print(f"Tiempo total (serial): {t:.2f} s")


if __name__ == '__main__':
    main()
