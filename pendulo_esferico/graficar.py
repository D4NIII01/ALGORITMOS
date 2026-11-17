import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def leer(ruta):
    datos = defaultdict(list)
    with open(ruta) as f:
        for fila in csv.DictReader(f):
            clave = (fila['version'], int(fila['n_trayectorias']), int(fila['procesos']))
            datos[clave].append(float(fila['tiempo_s']))
    return datos


def resumen(datos):
    return {k: (float(np.median(v)), min(v), max(v)) for k, v in datos.items()}


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--entrada', default=os.path.join(base, 'resultados', 'tiempos.csv'))
    p.add_argument('--salida', default=os.path.join(base, 'resultados'))
    a = p.parse_args()

    res = resumen(leer(a.entrada))
    ns = sorted({k[1] for k in res})
    procs = sorted({k[2] for k in res if k[0] == 'paralelo'})

    filas = []
    for n in ns:
        ts = res.get(('serial', n, 1))
        if ts is None:
            continue
        for pr in procs:
            tp = res.get(('paralelo', n, pr))
            if tp is None:
                continue
            s = ts[0] / tp[0]
            filas.append([n, pr, ts[0], tp[0], s, s / pr])

    with open(os.path.join(a.salida, 'resumen.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_trayectorias', 'procesos', 't_serial_s', 't_paralelo_s',
                    'speedup', 'eficiencia'])
        for fila in filas:
            w.writerow([fila[0], fila[1]] + [f"{x:.4f}" for x in fila[2:]])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [n for n in ns if ('serial', n, 1) in res]
    med = [res[('serial', n, 1)][0] for n in xs]
    err = [[res[('serial', n, 1)][0] - res[('serial', n, 1)][1] for n in xs],
           [res[('serial', n, 1)][2] - res[('serial', n, 1)][0] for n in xs]]
    ax.errorbar(xs, med, yerr=err, marker='o', capsize=3, label='serial')
    for pr in procs:
        xp = [n for n in ns if ('paralelo', n, pr) in res]
        if not xp:
            continue
        mp = [res[('paralelo', n, pr)][0] for n in xp]
        ep = [[res[('paralelo', n, pr)][0] - res[('paralelo', n, pr)][1] for n in xp],
              [res[('paralelo', n, pr)][2] - res[('paralelo', n, pr)][0] for n in xp]]
        ax.errorbar(xp, mp, yerr=ep, marker='s', capsize=3, label=f'paralelo, p={pr}')
    ax.set_xlabel('Numero de trayectorias')
    ax.set_ylabel('Tiempo (s)')
    ax.set_title('Tiempo de ejecucion (mediana, barras min-max)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.salida, 'fig1_tiempo_vs_n.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        pts = [(f[1], f[4]) for f in filas if f[0] == n]
        if pts:
            ax.plot(*zip(*pts), marker='o', label=f'n={n}')
    if procs:
        ax.plot([1, max(procs)], [1, max(procs)], 'k--', label='ideal')
    ax.set_xlabel('Numero de procesos')
    ax.set_ylabel('Speedup  T_serial / T_paralelo')
    ax.set_title('Aceleracion')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.salida, 'fig2_speedup.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        pts = [(f[1], f[5]) for f in filas if f[0] == n]
        if pts:
            ax.plot(*zip(*pts), marker='o', label=f'n={n}')
    ax.axhline(1.0, color='k', linestyle='--', label='ideal')
    ax.set_xlabel('Numero de procesos')
    ax.set_ylabel('Eficiencia  speedup / p')
    ax.set_title('Eficiencia paralela')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.salida, 'fig3_eficiencia.png'), dpi=200)
    plt.close(fig)

    print("Figuras y resumen.csv guardados en", a.salida)


if __name__ == '__main__':
    main()
