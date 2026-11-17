import modelo  # debe ir antes que numpy (fija los hilos de BLAS)
import argparse
import numpy as np
from serial import ejecutar_serial
from paralelo import ejecutar_paralelo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-lista', type=int, nargs='+', default=[2, 4, 8])
    ap.add_argument('--procesos-lista', type=int, nargs='+', default=[2, 4])
    ap.add_argument('--tfinal', type=float, default=40.0)
    ap.add_argument('--npuntos', type=int, default=1001)
    ap.add_argument('--semilla', type=int, default=0)
    a = ap.parse_args()

    print("n  p  convergidas_iguales  max|dif|   allclose  deriva_E_max")
    for n in a.n_lista:
        s = ejecutar_serial(n, a.tfinal, a.npuntos, a.semilla)
        for p in a.procesos_lista:
            if p > n:
                continue
            q = ejecutar_paralelo(n, p, a.tfinal, a.npuntos, a.semilla)
            ambos = [(x[0], y[0]) for x, y in zip(s, q) if x[2] and y[2]]
            dif = max(np.max(np.abs(x - y)) for x, y in ambos)
            ok = all(np.allclose(x, y, rtol=1e-8, atol=1e-8) for x, y in ambos)
            same = [x[2] for x in s] == [y[2] for y in q]
            dmax = max(x[1] for x in s if x[2])
            print(f"{n:<2d} {p:<2d} {str(same):<20s} {dif:<10.2e} {str(ok):<9s} {dmax:.2e}", flush=True)


if __name__ == '__main__':
    main()
