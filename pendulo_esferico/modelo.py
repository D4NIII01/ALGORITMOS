import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import sympy as sp
from scipy.integrate import odeint

G, M1, M2, LEN1, LEN2 = 9.81, 1, 200, 2, 1

S0_PRINCIPAL = [0, 1.0, 1.6, 2.2, 0.8, -0.5, 0, 0.3]


def construir_modelo():
    t, g = sp.symbols('t g')
    m1, m2, L1, L2 = sp.symbols('m1 m2 L1 L2')
    th1, th2, ph1, ph2 = [sp.Function(n)(t) for n in ('theta_1', 'theta_2', 'phi_1', 'phi_2')]
    q = [th1, th2, ph1, ph2]

    x1 = L1 * sp.cos(th1) * sp.sin(ph1)
    y1 = L1 * sp.sin(th1) * sp.sin(ph1)
    z1 = -L1 * sp.cos(ph1)
    x2 = x1 + L2 * sp.cos(th2) * sp.sin(ph2)
    y2 = y1 + L2 * sp.sin(th2) * sp.sin(ph2)
    z2 = z1 - L2 * sp.cos(ph2)

    T = sp.Rational(1, 2) * (m1 * (x1.diff(t)**2 + y1.diff(t)**2 + z1.diff(t)**2)
                             + m2 * (x2.diff(t)**2 + y2.diff(t)**2 + z2.diff(t)**2))
    V = g * (m1 * z1 + m2 * z2)
    Lag = T - V
    EL = [sp.diff(Lag, qi) - sp.diff(sp.diff(Lag, qi.diff(t)), t) for qi in q]

    a1, a2, p1, p2 = sp.symbols('a1 a2 p1 p2')
    va1, va2, vp1, vp2 = sp.symbols('va1 va2 vp1 vp2')
    aa1, aa2, ap1, ap2 = sp.symbols('aa1 aa2 ap1 ap2')

    subs = {}
    for f, acc in zip(q, (aa1, aa2, ap1, ap2)):
        subs[f.diff(t, 2)] = acc
    for f, vel in zip(q, (va1, va2, vp1, vp2)):
        subs[f.diff(t)] = vel
    for f, ang in zip(q, (a1, a2, p1, p2)):
        subs[f] = ang
    EL = [e.subs(subs) for e in EL]

    M, F = sp.linear_eq_to_matrix(EL, [aa1, aa2, ap1, ap2])
    args = (g, m1, m2, L1, L2, a1, a2, p1, p2, va1, va2, vp1, vp2)
    M_f = sp.lambdify(args, M, 'numpy')
    F_f = sp.lambdify(args, F, 'numpy')
    E_f = sp.lambdify(args, (T + V).subs(subs), 'numpy')
    pos_f = sp.lambdify((a1, a2, p1, p2, L1, L2),
                        [e.subs(subs) for e in (x1, y1, z1, x2, y2, z2)], 'numpy')
    return M_f, F_f, E_f, pos_f


def integrar(S0, modelo, tfinal=40.0, npuntos=1001):
    M_f, F_f, E_f, pos_f = modelo
    t_eval = np.linspace(0, tfinal, npuntos)

    def dSdt(S, t_):
        a1, a2, p1, p2, va1, va2, vp1, vp2 = S
        args = (G, M1, M2, LEN1, LEN2, a1, a2, p1, p2, va1, va2, vp1, vp2)
        acc = np.linalg.solve(np.array(M_f(*args), dtype=float),
                              np.array(F_f(*args), dtype=float)).ravel()
        return [va1, va2, vp1, vp2, *acc]

    try:
        ans, info = odeint(dSdt, y0=S0, t=t_eval, rtol=1e-10, atol=1e-10,
                           mxstep=50000, full_output=True)
        ok = bool(info['message'] == 'Integration successful.' and np.isfinite(ans).all())
    except Exception:
        return None, float('nan'), False

    E = E_f(G, M1, M2, LEN1, LEN2, *ans.T)
    deriva = float(np.max(np.abs((E - E[0]) / E[0])))
    xyz = np.array(pos_f(ans[:, 0], ans[:, 1], ans[:, 2], ans[:, 3], LEN1, LEN2))
    return xyz, deriva, ok


def condiciones_iniciales(n, semilla=0):
    rng = np.random.default_rng(semilla)
    lista = [np.array(S0_PRINCIPAL, dtype=float)]
    for _ in range(n - 1):
        th = rng.uniform(0, 2 * np.pi, 2)
        ph = rng.uniform(0.4, np.pi - 0.4, 2)
        vel = rng.uniform(-1, 1, 4)
        lista.append(np.concatenate([th, ph, vel]))
    return lista


def guardar(resultados, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    buenas = [r[0] for r in resultados if r[2]]
    if not buenas:
        raise RuntimeError("Ninguna trayectoria convergio")
    principal = resultados[0][0] if resultados[0][2] else buenas[0]
    np.save(os.path.join(carpeta, '3Dpen.npy'), principal)
    np.save(os.path.join(carpeta, '3Dpen_all.npy'), np.array(buenas))
    return len(buenas)
