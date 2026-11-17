"""Perfilado de una trayectoria (la principal): cuanto cuesta construir el modelo
y como se reparte el tiempo de integracion.
Uso: python perfilar.py [--tfinal 5]"""
import modelo  # debe ir antes que numpy (fija los hilos de BLAS)
import argparse
import cProfile
import pstats
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfinal', type=float, default=5.0)
    ap.add_argument('--npuntos', type=int, default=126)
    a = ap.parse_args()

    t0 = time.perf_counter()
    mod = modelo.construir_modelo()
    t_modelo = time.perf_counter() - t0
    S0 = modelo.condiciones_iniciales(1, 0)[0]

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    _, deriva, ok = modelo.integrar(S0, mod, a.tfinal, a.npuntos)
    pr.disable()
    t_int = time.perf_counter() - t0

    st = pstats.Stats(pr).stats  # (archivo, linea, nombre) -> (cc, nc, tt, ct, callers)
    lam = sorted([v for k, v in st.items() if k[0].startswith('<lambdifygenerated')],
                 key=lambda v: v[3], reverse=True)
    dsdt = [v for k, v in st.items() if k[2] == 'dSdt']
    solve = [v for k, v in st.items() if k[2] == 'solve']
    pct = lambda v: 100 * v[3] / t_int

    print(f"construir_modelo: {t_modelo:.2f} s")
    print(f"integrar ({a.tfinal:g} s simulados): {t_int:.2f} s, convergio={ok}, deriva={deriva:.2e}")
    if dsdt:
        print(f"evaluaciones de dSdt: {dsdt[0][1]}, {pct(dsdt[0]):.0f} % del tiempo de integracion")
    if len(lam) >= 2:
        print(f"funcion generada para F: {pct(lam[0]):.0f} %")
        print(f"funcion generada para M: {pct(lam[1]):.0f} %")
    if solve:
        print(f"np.linalg.solve: {pct(solve[0]):.0f} %")


if __name__ == '__main__':
    main()
