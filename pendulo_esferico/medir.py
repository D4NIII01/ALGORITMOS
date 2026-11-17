import argparse
import csv
import json
import os
import platform
import sys
import time
from multiprocessing import cpu_count

import numpy as np
import scipy
import sympy

from serial import ejecutar_serial
from paralelo import ejecutar_paralelo

COLUMNAS = ['version', 'n_trayectorias', 'procesos', 'repeticion',
            'tiempo_s', 'n_convergidas', 'tfinal', 'npuntos']


def modelo_cpu():
    try:
        with open('/proc/cpuinfo') as f:
            for linea in f:
                if linea.startswith('model name'):
                    return linea.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def memoria_gb():
    try:
        with open('/proc/meminfo') as f:
            for linea in f:
                if linea.startswith('MemTotal'):
                    return round(int(linea.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    return None


def info_hardware():
    return {
        'sistema': platform.platform(),
        'cpu': modelo_cpu(),
        'nucleos_logicos': cpu_count(),
        'memoria_gb': memoria_gb(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'sympy': sympy.__version__,
    }


def procesos_por_defecto():
    n = cpu_count()
    lista, p = [], 1
    while p < n:
        lista.append(p)
        p *= 2
    lista.append(n)
    return sorted(set(lista))


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--n-lista', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    p.add_argument('--procesos-lista', type=int, nargs='+', default=procesos_por_defecto())
    p.add_argument('--repeticiones', type=int, default=5)
    p.add_argument('--tfinal', type=float, default=40.0)
    p.add_argument('--npuntos', type=int, default=1001)
    p.add_argument('--semilla', type=int, default=0)
    p.add_argument('--salida', default=os.path.join(base, 'resultados', 'tiempos.csv'))
    a = p.parse_args()

    carpeta = os.path.dirname(os.path.abspath(a.salida))
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, 'hardware.json'), 'w') as f:
        json.dump(info_hardware(), f, indent=2)

    with open(a.salida, 'w', newline='') as f:
        escritor = csv.writer(f)
        escritor.writerow(COLUMNAS)

        def registrar(version, n, procs, rep, t, resultados):
            ok = sum(1 for r in resultados if r[2])
            escritor.writerow([version, n, procs, rep, f"{t:.6f}", ok, a.tfinal, a.npuntos])
            f.flush()
            print(f"{version:9s} n={n:<3d} procesos={procs:<3d} rep={rep} "
                  f"t={t:8.2f} s  convergidas={ok}/{n}", flush=True)

        for rep in range(1, a.repeticiones + 1):
            for n in a.n_lista:
                t0 = time.perf_counter()
                res = ejecutar_serial(n, a.tfinal, a.npuntos, a.semilla)
                registrar('serial', n, 1, rep, time.perf_counter() - t0, res)

                for procs in a.procesos_lista:
                    if procs > n:
                        continue
                    t0 = time.perf_counter()
                    res = ejecutar_paralelo(n, procs, a.tfinal, a.npuntos, a.semilla)
                    registrar('paralelo', n, procs, rep, time.perf_counter() - t0, res)

    print("Resultados guardados en", a.salida)


if __name__ == '__main__':
    main()
