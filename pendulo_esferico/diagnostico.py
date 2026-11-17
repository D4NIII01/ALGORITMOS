"""Diagnostico de la trayectoria principal: energia inicial, deriva de energia,
rangos de los angulos, recorrido de la masa 2 y figuras (3D y energia).
Uso: python diagnostico.py [--tfinal 40] [--salida resultados]"""
import modelo  # debe ir antes que numpy (fija los hilos de BLAS)
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfinal', type=float, default=40.0)
    ap.add_argument('--npuntos', type=int, default=1001)
    ap.add_argument('--salida', default=os.path.join(base, 'resultados'))
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)

    M_f, F_f, E_f, pos_f = modelo.construir_modelo()
    G, M1, M2, L1, L2 = modelo.G, modelo.M1, modelo.M2, modelo.LEN1, modelo.LEN2

    def dSdt(S, t_):
        a1, a2, p1, p2, v1, v2, w1, w2 = S
        args = (G, M1, M2, L1, L2, a1, a2, p1, p2, v1, v2, w1, w2)
        acc = np.linalg.solve(np.array(M_f(*args), float), np.array(F_f(*args), float)).ravel()
        return [v1, v2, w1, w2, *acc]

    te = np.linspace(0, a.tfinal, a.npuntos)
    ans, info = odeint(dSdt, modelo.S0_PRINCIPAL, te, rtol=1e-10, atol=1e-10,
                       mxstep=50000, full_output=True)
    E = E_f(G, M1, M2, L1, L2, *ans.T)
    deriva = (E - E[0]) / E[0]
    x1, y1, z1, x2, y2, z2 = np.array(pos_f(ans[:, 0], ans[:, 1], ans[:, 2], ans[:, 3], L1, L2))

    lineas = [
        f"S0 = {list(modelo.S0_PRINCIPAL)}",
        f"Integracion: {info['message']}",
        f"E(0) = {E[0]:.2f} J",
        f"Deriva relativa maxima de energia = {np.max(np.abs(deriva)):.2e}",
        f"phi1 en [{ans[:, 2].min():.2f}, {ans[:, 2].max():.2f}] rad",
        f"phi2 en [{ans[:, 3].min():.2f}, {ans[:, 3].max():.2f}] rad",
        f"Recorrido de la masa 2 (x, y, z) = {np.ptp(x2):.1f} x {np.ptp(y2):.1f} x {np.ptp(z2):.1f} m",
    ]
    print("\n".join(lineas))
    with open(os.path.join(a.salida, 'diagnostico.txt'), 'w') as f:
        f.write("\n".join(lineas) + "\n")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    ax.plot(x1, y1, z1, lw=.6, label=f'masa 1 (m1 = {M1} kg)')
    ax.plot(x2, y2, z2, lw=.6, label=f'masa 2 (m2 = {M2} kg)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.legend(); ax.set_title(f'Trayectoria principal, 0-{a.tfinal:g} s')
    fig.tight_layout(); fig.savefig(os.path.join(a.salida, 'fig_trayectoria.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(te, deriva)
    ax.set_xlabel('t (s)'); ax.set_ylabel('(E - E0)/E0'); ax.grid(alpha=.3)
    ax.set_title('Deriva relativa de energia')
    fig.tight_layout(); fig.savefig(os.path.join(a.salida, 'fig_energia.png'), dpi=200)
    plt.close(fig)
    print("Figuras y diagnostico.txt guardados en", a.salida)


if __name__ == '__main__':
    main()
