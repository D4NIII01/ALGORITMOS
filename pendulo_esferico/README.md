# Péndulo esférico doble: versión serial y paralela

Proyecto final de Algoritmos y Arquitecturas Computacionales de Alto Rendimiento (Universidad Industrial de Santander, 2026-2).

Simulación de un péndulo esférico doble (mecánica lagrangiana con SymPy, integración con `scipy.integrate.odeint`) y medición del rendimiento de una versión serial frente a una paralela con `multiprocessing`.

La unidad de trabajo es la integración de **una trayectoria** (una condición inicial). Una sola trayectoria es secuencial por naturaleza, así que la paralelización consiste en resolver **n trayectorias independientes** repartidas entre `p` procesos. La primera trayectoria usa la condición inicial principal; las demás son aleatorias y reproducibles mediante una semilla.

## Contenido

| Archivo | Descripción |
|---|---|
| `modelo.py` | Código común: modelo simbólico, integración de una trayectoria, condiciones iniciales y guardado. Lo usan todas las versiones. |
| `serial.py` | Versión serial: resuelve las n trayectorias una tras otra. |
| `paralelo.py` | Versión paralela: reparte las n trayectorias entre `p` procesos (`multiprocessing.Pool`). |
| `verificar.py` | Compara serial y paralelo (diferencia máxima, `np.allclose` y deriva de energía) para varios `n` y `p`. |
| `perfilar.py` | Perfilado con `cProfile` de una trayectoria: costo de construir el modelo y reparto del tiempo de integración. |
| `diagnostico.py` | Diagnóstico de la trayectoria principal: energía inicial, deriva de energía, rangos de φ, recorrido de la masa 2 y las figuras 3D y de energía. |
| `medir.py` | Ejecuta los experimentos de tiempo y los guarda en `resultados/tiempos.csv`. También escribe `resultados/hardware.json`. |
| `graficar.py` | Lee el CSV y genera las figuras de desempeño y `resultados/resumen.csv`. |
| `animacion.py` | Animación 3D con VPython a partir de `data/3Dpen.npy`. |
| `requirements.txt` | Dependencias (la animación necesita además `vpython`, ver más abajo). |

## Entorno

- **Python:** 3.14.9 (con el que se tomaron las mediciones). También se verificó con 3.12.3; debería funcionar con 3.10 o superior.
- **Librerías** (`requirements.txt`): `numpy>=2.0`, `scipy>=1.13`, `sympy>=1.12`, `matplotlib>=3.8`.
  Versiones de las mediciones: numpy 2.5.4, scipy 1.18.1, sympy 1.14.0.
- **Solo para la animación:** `vpython` (no está en `requirements.txt` para que el resto funcione sin navegador).

### Hardware usado en las mediciones

`medir.py` guarda estos datos automáticamente en `resultados/hardware.json`.

| Elemento | Valor |
|---|---|
| CPU | 12th Gen Intel Core i5-1235U (15 W) |
| Núcleos | 10 físicos y 12 hilos lógicos (según el fabricante: 2 de rendimiento con 2 hilos y 8 de eficiencia con 1 hilo) |
| Frecuencia | 0,4 a 4,4 GHz |
| Caché | L2 6,5 MiB, L3 12 MiB |
| RAM | 15,3 GB |
| Sistema | Linux 7.3.3 (CachyOS) |

La CPU es híbrida: los núcleos de eficiencia son más lentos que los de rendimiento, y esto afecta al speedup (ver el informe, sección 6).

## Instalación

Desde la carpeta del proyecto:

```
python -m venv venv
source venv/bin/activate        # bash/zsh
source venv/bin/activate.fish   # fish
venv\Scripts\activate           # Windows
pip install -r requirements.txt
pip install vpython             # solo para animacion.py
```

El entorno hay que activarlo **en cada terminal nueva**. Si aparece `No module named 'scipy'` es que se está usando el Python del sistema y no el del entorno. Los comandos de este README se ejecutan desde la carpeta del proyecto (las rutas de `resultados/` y `data/` se calculan a partir de la ubicación de cada script).

## Uso

Serial (8 trayectorias, 40 s simulados cada una):

```
python serial.py --n 8
```

Paralelo (8 trayectorias en 4 procesos):

```
python paralelo.py --n 8 --procesos 4
```

Ambos guardan en `data/`: `3Dpen.npy` (primera trayectoria, forma `(6, npuntos)` con filas `x1, y1, z1, x2, y2, z2`) y `3Dpen_all.npy` (todas las convergidas, forma `(n, 6, npuntos)`), e imprimen la deriva relativa de energía de cada trayectoria y el tiempo total.

Opciones comunes: `--tfinal` (tiempo simulado, 40 por defecto), `--npuntos` (1001), `--semilla` (0) y `--salida` (carpeta de salida).

### Parámetros y condición inicial principal

Se configuran en `modelo.py`:

| Parámetro | Valor |
|---|---|
| g | 9,81 m/s² |
| m1, m2 | 1 kg, 10 kg |
| L1, L2 | 2 m, 1 m |
| `S0_PRINCIPAL` = [θ1, θ2, φ1, φ2, θ̇1, θ̇2, φ̇1, φ̇2] | [0, 1,0, 1,6, 2,2, 0,8, −0,5, 0, 0,3] (rad y rad/s) |

Los ángulos φ se miden desde la vertical descendente: φ = 0 es el punto más bajo. Si se cambia `S0_PRINCIPAL`, hay que volver a ejecutar `diagnostico.py`, `perfilar.py` y `verificar.py`, y regenerar la animación y las mediciones.

### Animación

```
python serial.py --n 1      # genera data/3Dpen.npy
python animacion.py         # abre la animación en el navegador
```

La animación consume CPU, así que **no debe ejecutarse mientras corre `medir.py`**: alteraría los tiempos. Para mostrarla durante una medición conviene grabar antes un video de la pantalla.

## Reproducir los resultados del informe

Orden recomendado (los pasos cortos primero y `medir.py` al final, con el equipo libre y conectado a la corriente):

| # | Comando | Qué genera | Sección del informe |
|---|---|---|---|
| 1 | `python serial.py --n 1` | `data/3Dpen.npy` para la animación | — |
| 2 | `python diagnostico.py` | `resultados/diagnostico.txt`, `fig_trayectoria.png`, `fig_energia.png` | 1 y 2 |
| 3 | `python perfilar.py` | Costo del modelo y reparto del tiempo de integración (salida por pantalla) | 3 |
| 4 | `lscpu` | Núcleos y frecuencias de la CPU (Linux) | 5 |
| 5 | `python verificar.py --tfinal 40 --n-lista 2 4 8 16` | Tabla serial vs paralelo (por pantalla) | 4 |
| 6 | `python medir.py` | `resultados/tiempos.csv` y `resultados/hardware.json` | 5 |
| 7 | `python graficar.py` | `resultados/resumen.csv`, `fig1_tiempo_vs_n.png`, `fig2_speedup.png`, `fig3_eficiencia.png` | 5 y 6 |

Para guardar la salida de un script además de verla en pantalla: `python perfilar.py | tee resultados/perfilado.txt`.

### Valores por defecto de `medir.py`

Por cada repetición, cada `n` y cada `p`, `medir.py` ejecuta una corrida serial y una paralela y escribe una fila por corrida en `resultados/tiempos.csv` con las columnas `version, n_trayectorias, procesos, repeticion, tiempo_s, n_convergidas, tfinal, npuntos`.

- `n` en {1, 2, 4, 8, 16}
- `p` en potencias de 2 menores que el número de núcleos lógicos, más ese número (en este equipo: 1, 2, 4, 8 y 12); se omiten los `p` mayores que `n`
- 5 repeticiones

Con estos valores el CSV tiene 101 líneas (100 mediciones más el encabezado) y la ejecución dura alrededor de una hora. Para una prueba rápida:

```
python medir.py --n-lista 1 2 --procesos-lista 1 2 --repeticiones 2 --tfinal 3 --salida resultados/prueba.csv
```

Parámetros de `medir.py`: `--n-lista`, `--procesos-lista`, `--repeticiones`, `--tfinal`, `--npuntos`, `--semilla` y `--salida`.

**Atención:** `medir.py` sobrescribe el archivo de salida. Para no perder una medición larga, use `--salida` en las pruebas o copie `resultados/` antes de volver a ejecutarlo.

Si ya existe `tiempos.csv`, basta con volver a ejecutar `python graficar.py` para regenerar las figuras sin repetir las mediciones.

| Figura | Archivo | Qué muestra |
|---|---|---|
| Fenómeno | `resultados/fig_trayectoria.png` | Trayectorias 3D de las dos masas durante 40 s. |
| Energía | `resultados/fig_energia.png` | Deriva relativa de la energía total. |
| 1 | `resultados/fig1_tiempo_vs_n.png` | Tiempo de ejecución frente al número de trayectorias, para el serial y para cada `p`. Mediana con barras mín–máx. |
| 2 | `resultados/fig2_speedup.png` | Speedup `T_serial / T_paralelo` frente al número de procesos, una curva por `n`, con la recta ideal. |
| 3 | `resultados/fig3_eficiencia.png` | Eficiencia `speedup / p` frente al número de procesos. |

`graficar.py` también escribe `resultados/resumen.csv` con la tabla de tiempos, speedup y eficiencia por `(n, p)`.

## Resultados obtenidos

Mediana de 5 repeticiones en el equipo descrito arriba (detalle completo en `resultados/resumen.csv`).

| n | p | T serial (s) | T paralelo (s) | Speedup | Eficiencia |
|---|---|---|---|---|---|
| 16 | 1 | 123,80 | 116,45 | 1,06 | 1,06 |
| 16 | 2 | 123,80 | 75,39 | 1,64 | 0,82 |
| 16 | 4 | 123,80 | 57,09 | 2,17 | 0,54 |
| 16 | 8 | 123,80 | 41,78 | 2,96 | 0,37 |
| 16 | 12 | 123,80 | 40,76 | 3,04 | 0,25 |

- **Verificación:** la diferencia máxima entre serial y paralelo es 0 y `np.allclose` es verdadero para n = 2, 4, 8 y 16 con p = 2 y 4.
- **Física:** para la trayectoria principal, E(0) = 73,50 J y la deriva relativa máxima de energía en 40 s es 6,04·10⁻⁸.
- **Perfilado:** construir el modelo cuesta 0,65 s; el 99 % del tiempo de integración está en `dSdt` (la función de F pesa 76 %, la de M 15 % y `np.linalg.solve` 6 %).
- **Interpretación:** con p = 1 el paralelo iguala al serial, así que el sobrecosto de crear procesos es despreciable. La fracción paralelizable efectiva que explica los datos es f ≈ 0,74, frente a ≈ 0,995 que predice el código; la diferencia se atribuye al hardware (CPU híbrida de 15 W), según la discusión del informe.

## Notas metodológicas

- **Tiempo medido:** el tiempo de cada corrida incluye todo el trabajo necesario: construcción del modelo simbólico, integración de todas las trayectorias y, en el caso paralelo, la creación del `Pool`. En el paralelo cada proceso construye su propio modelo, porque las funciones generadas por `lambdify` no se pueden enviar entre procesos (unos 0,65 s por proceso, en paralelo).
- **Reloj:** `time.perf_counter()`.
- **Repeticiones:** se reporta la mediana y el rango mín–máx de las repeticiones.
- **Condiciones iniciales:** la primera trayectoria usa `S0_PRINCIPAL`; las demás son aleatorias (θ ~ U(0, 2π), φ ~ U(0,4; π − 0,4) rad, cuatro velocidades ~ U(−1, 1)). La semilla es la misma en todas las corridas, así que serial y paralelo resuelven exactamente las mismas trayectorias.
- **Hilos de BLAS/OpenMP:** `modelo.py` fija `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` y `MKL_NUM_THREADS` en 1 (si no están definidas) antes de importar NumPy, para que los procesos no compitan por los núcleos con hilos internos de las librerías. Los módulos que se ejecutan deben importar `modelo` antes que NumPy, como hacen `serial.py`, `paralelo.py`, `medir.py`, `verificar.py`, `perfilar.py` y `diagnostico.py`.
- **Balance de carga:** las trayectorias no cuestan lo mismo. `pool.map(..., chunksize=1)` reparte una trayectoria por tarea para reducir el desbalance, pero con pocas trayectorias la eficiencia puede ser baja.
- **Validación numérica:** cada trayectoria reporta la deriva relativa de la energía total. Una trayectoria se marca como fallida si `odeint` no converge o produce valores no finitos, y no se incluye en los archivos guardados.
- **Limitación física:** con estas coordenadas, φ = 0 es el punto más bajo y la matriz de masa M se vuelve singular cuando `sen φ = 0` (φ = 0 y φ = π). Las condiciones iniciales aleatorias mantienen φ entre 0,4 y π − 0,4 rad, pero una trayectoria caótica puede acercarse a esos puntos más tarde (en la trayectoria principal, φ1 llega a 0,03 rad y φ2 a 0,04 rad del punto más bajo).
- **Método de arranque de procesos:** `multiprocessing` usa `spawn` en Windows y macOS. Desde Python 3.14 el valor por defecto en Linux tampoco es `fork` (pasó a `forkserver`); consulte la documentación de su versión. El código funciona igual con cualquiera de ellos, porque cada proceso construye su modelo en el inicializador del `Pool` y los scripts tienen la protección `if __name__ == '__main__'`. Por eso los scripts deben ejecutarse desde la línea de comandos y no desde una celda interactiva.

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `ModuleNotFoundError: No module named 'scipy'` | No está activado el entorno virtual. Actívelo (con `activate.fish` si usa fish) o llame al Python del entorno directamente. |
| `source ../venv/bin/activate` falla en fish | El script `activate` es para bash. Use `activate.fish`. |
| `FileNotFoundError` con `3Dpen.npy` | Falta generar los datos: `python serial.py --n 1`. |
| `tiempos.csv` quedó con pocas filas | Una ejecución posterior de `medir.py` (por ejemplo, una prueba rápida) lo sobrescribió. Repita la medición completa. |
| Las figuras no coinciden con `tiempos.csv` | Vuelva a ejecutar `python graficar.py`. |
| La animación no abre | Instale `vpython` (`pip install vpython`). El aviso sobre `pkg_resources` es solo una advertencia. |
