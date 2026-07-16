"""
=============================================================================
  ANALIZADOR DE PATRONES - LOTO HISTÓRICA
  Busca algoritmos que predigan lo que ya sucedió, en el orden en que ocurrió.
  Backtesting walk-forward sobre los 525 sorteos del historico_loto.csv
=============================================================================
"""

import csv
import os
import math
import itertools
import numpy as np
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_datos(path='c:/Users/Administrador/Desktop/lot/historico_loto.csv'):
    sorteos = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nums = sorted([int(row[f'B{i}']) for i in range(1, 7)])
                sorteos.append({
                    'sorteo': int(row['Sorteo']),
                    'fecha':  row['Fecha'],
                    'nums':   nums,
                    'plus':   int(row['Plus'])
                })
            except Exception:
                continue
    # El CSV ya está en orden cronológico (ascendente)
    return sorteos

# ─────────────────────────────────────────────────────────────────────────────
# 2. UTILIDADES ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────────────────────

MAX_NUM = 45

def frecuencias_ventana(sorteos, ventana=20):
    """Frecuencia relativa de cada número en los últimos `ventana` sorteos."""
    freq = {n: 0.0 for n in range(MAX_NUM + 1)}
    recientes = sorteos[-ventana:]
    total = len(recientes) * 6
    for s in recientes:
        for n in s['nums']:
            freq[n] += 1.0
    if total > 0:
        for n in freq:
            freq[n] /= total
    return freq

def delays_actuales(sorteos):
    """Cuántos sorteos hace que cada número no salió."""
    last_seen = {}
    for i, s in enumerate(sorteos):
        for n in s['nums']:
            last_seen[n] = i
    n_total = len(sorteos)
    return {n: n_total - 1 - last_seen.get(n, -1) if n in last_seen else n_total
            for n in range(MAX_NUM + 1)}

def matriz_coocurrencia(sorteos):
    cooc = defaultdict(lambda: defaultdict(int))
    for s in sorteos:
        for a, b in itertools.combinations(s['nums'], 2):
            cooc[a][b] += 1
            cooc[b][a] += 1
    return cooc

def matriz_markov(sorteos):
    """P(n aparece en t+1 | m apareció en t)."""
    trans = defaultdict(lambda: defaultdict(int))
    for i in range(1, len(sorteos)):
        for prev in sorteos[i-1]['nums']:
            for curr in sorteos[i]['nums']:
                trans[prev][curr] += 1
    return trans

def ewma(series, span):
    alpha = 2.0 / (span + 1)
    val = series[0] if len(series) > 0 else 0.0
    for x in series[1:]:
        val = alpha * x + (1 - alpha) * val
    return val

def pares_impares(nums):
    pares = sum(1 for n in nums if n % 2 == 0)
    return pares, 6 - pares

def suma_total(nums):
    return sum(nums)

def spread(nums):
    return max(nums) - min(nums)

def decenas(nums):
    d = defaultdict(int)
    for n in nums:
        d[n // 10] += 1
    return dict(d)

def consecutivos(nums):
    s = sorted(nums)
    return sum(1 for i in range(1, len(s)) if s[i] - s[i-1] == 1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. ALGORITMOS DE PREDICCIÓN (score para cada número 0..45)
# ─────────────────────────────────────────────────────────────────────────────

def score_frecuencia(hist):
    """Números más frecuentes en las últimas 20 jugadas."""
    return frecuencias_ventana(hist, ventana=20)

def score_delay_geometrico(hist):
    """
    Modelo geométrico de retorno: prob = (1/mean_delay) * (1 - 1/mean_delay)^delay
    Cuanto mayor el delay frente a la media histórica, mayor la probabilidad de volver.
    """
    delays_hist = defaultdict(list)
    for i, s in enumerate(hist):
        for n in s['nums']:
            if n in delays_hist and len(delays_hist[n]) > 0:
                last = delays_hist[n][-1]
                # guardamos el índice, calculamos gaps después
            delays_hist[n].append(i)

    # Recalcular gaps reales
    gaps = defaultdict(list)
    for n, indices in delays_hist.items():
        for j in range(1, len(indices)):
            gaps[n].append(indices[j] - indices[j-1])

    delay_actual = delays_actuales(hist)
    score = {}
    for n in range(MAX_NUM + 1):
        g = gaps.get(n, [])
        mean_g = np.mean(g) if g else 7.6
        d = delay_actual[n]
        # P(retorno en este sorteo | delay actual)
        if mean_g > 0:
            p = (1.0 / mean_g) * ((1 - 1.0 / mean_g) ** max(d - 1, 0))
        else:
            p = 0.0
        score[n] = p
    return score

def score_markov(hist):
    """Probabilidad de Markov orden 1: basada en el último sorteo."""
    if len(hist) < 2:
        return {n: 1/46 for n in range(MAX_NUM + 1)}
    trans = matriz_markov(hist)
    ultimo = hist[-1]['nums']
    score = defaultdict(float)
    for prev in ultimo:
        total = sum(trans[prev].values())
        if total > 0:
            for nxt, cnt in trans[prev].items():
                score[nxt] += cnt / total
    # Normalizar
    total_score = sum(score.values()) or 1
    return {n: score[n] / total_score for n in range(MAX_NUM + 1)}

def score_coocurrencia(hist):
    """Números que más co-ocurren con los del último sorteo."""
    if len(hist) < 2:
        return {n: 1/46 for n in range(MAX_NUM + 1)}
    cooc = matriz_coocurrencia(hist)
    ultimo = set(hist[-1]['nums'])
    score = defaultdict(float)
    for n in range(MAX_NUM + 1):
        if n in ultimo:
            continue
        for u in ultimo:
            score[n] += cooc[n].get(u, 0)
    total = sum(score.values()) or 1
    return {n: score[n] / total for n in range(MAX_NUM + 1)}

def score_ewma_crossover(hist):
    """EWMA corto > EWMA largo → número 'caliente'."""
    score = {}
    for n in range(MAX_NUM + 1):
        serie = np.array([1.0 if n in s['nums'] else 0.0 for s in hist])
        if len(serie) < 20:
            score[n] = 0.5
            continue
        v3  = ewma(serie, 3)
        v10 = ewma(serie, 10)
        v30 = ewma(serie, 30)
        # Si cruza al alza: corto > largo
        cross = 1.0 if v3 > v30 else 0.0
        score[n] = 0.5 * v10 + 0.5 * cross
    return score

def score_fft_prediccion(hist):
    """
    Predicción por análisis espectral: proyecta el coseno de las 3 frecuencias
    dominantes al siguiente paso temporal.
    """
    n_t = len(hist)
    score = {}
    for n in range(MAX_NUM + 1):
        serie = np.array([1.0 if n in s['nums'] else 0.0 for s in hist])
        if n_t < 8:
            score[n] = float(np.mean(serie))
            continue
        sc = serie - np.mean(serie)
        fft_v = np.fft.fft(sc)
        freqs = np.fft.fftfreq(n_t)
        mitad = n_t // 2
        amps = np.abs(fft_v[:mitad])
        amps[0] = 0
        top = np.argsort(amps)[-3:]
        proyeccion = 0.0
        for idx in top:
            if amps[idx] > 0:
                fase = np.angle(fft_v[idx])
                proyeccion += amps[idx] * np.cos(2 * np.pi * freqs[idx] * n_t + fase)
        score[n] = float(np.mean(serie) + proyeccion / n_t)
    return score

def score_ensemble(hist, pesos=None):
    """
    Combina todos los algoritmos con pesos ajustables.
    pesos = dict con claves: 'freq', 'delay', 'markov', 'cooc', 'ewma', 'fft'
    """
    if pesos is None:
        pesos = {'freq': 0.20, 'delay': 0.20, 'markov': 0.20,
                 'cooc': 0.15, 'ewma': 0.15, 'fft': 0.10}

    s_freq   = score_frecuencia(hist)
    s_delay  = score_delay_geometrico(hist)
    s_markov = score_markov(hist)
    s_cooc   = score_coocurrencia(hist)
    s_ewma   = score_ewma_crossover(hist)
    s_fft    = score_fft_prediccion(hist)

    def normalizar(d):
        total = sum(d.values()) or 1
        return {k: v / total for k, v in d.items()}

    s_freq   = normalizar(s_freq)
    s_delay  = normalizar(s_delay)
    s_markov = normalizar(s_markov)
    s_cooc   = normalizar(s_cooc)
    s_ewma   = normalizar(s_ewma)
    s_fft    = normalizar(s_fft)

    combined = {}
    for n in range(MAX_NUM + 1):
        combined[n] = (
            pesos['freq']   * s_freq.get(n, 0) +
            pesos['delay']  * s_delay.get(n, 0) +
            pesos['markov'] * s_markov.get(n, 0) +
            pesos['cooc']   * s_cooc.get(n, 0) +
            pesos['ewma']   * s_ewma.get(n, 0) +
            pesos['fft']    * s_fft.get(n, 0)
        )
    return combined

# ─────────────────────────────────────────────────────────────────────────────
# 4. ANÁLISIS ESTADÍSTICO DEL HISTÓRICO (sin predicción)
# ─────────────────────────────────────────────────────────────────────────────

def analisis_estadistico_completo(sorteos):
    print("\n" + "="*60)
    print("  ANÁLISIS ESTADÍSTICO COMPLETO DEL HISTÓRICO")
    print(f"  Total sorteos: {len(sorteos)}  |  {sorteos[0]['fecha']} -> {sorteos[-1]['fecha']}")
    print("="*60)

    # 4.1 Frecuencia global
    freq_global = Counter()
    for s in sorteos:
        freq_global.update(s['nums'])

    print("\n[1] TOP 10 NÚMEROS MÁS FRECUENTES:")
    for num, cnt in freq_global.most_common(10):
        pct = cnt / len(sorteos) * 100 / 6 * 6  # apariciones / total bolas posibles
        bar = "#" * int(cnt / len(sorteos) * 100)
        print(f"  Nº {num:02d}: {cnt:3d} veces ({cnt/len(sorteos)*100:.1f}%) {bar}")

    print("\n[2] TOP 10 NÚMEROS MENOS FRECUENTES (más fríos histórico):")
    for num, cnt in freq_global.most_common()[:-11:-1]:
        print(f"  Nº {num:02d}: {cnt:3d} veces ({cnt/len(sorteos)*100:.1f}%)")

    # 4.2 Patrones de suma
    sumas = [suma_total(s['nums']) for s in sorteos]
    print(f"\n[3] SUMAS DE LOS 6 NÚMEROS:")
    print(f"  Media: {np.mean(sumas):.1f} | Mediana: {np.median(sumas):.0f} | Mín: {min(sumas)} | Máx: {max(sumas)}")
    print(f"  El 80% de las sumas cae entre {np.percentile(sumas,10):.0f} y {np.percentile(sumas,90):.0f}")

    # 4.3 Paridad
    parity = [pares_impares(s['nums'])[0] for s in sorteos]
    cnt_parity = Counter(parity)
    print(f"\n[4] DISTRIBUCIÓN PARIDAD (número de pares en el sorteo):")
    for k in sorted(cnt_parity.keys()):
        bar = "#" * int(cnt_parity[k] / len(sorteos) * 100)
        print(f"  {k} pares: {cnt_parity[k]:3d} veces ({cnt_parity[k]/len(sorteos)*100:.1f}%) {bar}")

    # 4.4 Spread
    spreads = [spread(s['nums']) for s in sorteos]
    print(f"\n[5] SPREAD (máx - mín):")
    print(f"  Media: {np.mean(spreads):.1f} | P10: {np.percentile(spreads,10):.0f} | P90: {np.percentile(spreads,90):.0f}")

    # 4.5 Pares más frecuentes
    print(f"\n[6] TOP 15 PARES DE NÚMEROS MÁS FRECUENTES (co-ocurrencia):")
    cooc_global = Counter()
    for s in sorteos:
        for a, b in itertools.combinations(s['nums'], 2):
            cooc_global[(a, b)] += 1
    for par, cnt in cooc_global.most_common(15):
        print(f"  {par[0]:02d}-{par[1]:02d}: {cnt} veces ({cnt/len(sorteos)*100:.1f}% de sorteos)")

    # 4.6 Tripletas más frecuentes
    print(f"\n[7] TOP 10 TRIPLETAS MÁS FRECUENTES:")
    trips = Counter()
    for s in sorteos:
        for t in itertools.combinations(s['nums'], 3):
            trips[t] += 1
    for tri, cnt in trips.most_common(10):
        print(f"  {tri[0]:02d}-{tri[1]:02d}-{tri[2]:02d}: {cnt} veces")

    # 4.7 Overlap entre sorteos consecutivos
    overlaps = []
    for i in range(1, len(sorteos)):
        ov = len(set(sorteos[i]['nums']).intersection(sorteos[i-1]['nums']))
        overlaps.append(ov)
    cnt_ov = Counter(overlaps)
    print(f"\n[8] SUPERPOSICIÓN CON EL SORTEO ANTERIOR (cuántos números repiten):")
    for k in sorted(cnt_ov.keys()):
        bar = "#" * int(cnt_ov[k] / len(overlaps) * 100)
        print(f"  {k} números en común: {cnt_ov[k]:3d} ({cnt_ov[k]/len(overlaps)*100:.1f}%) {bar}")

    # 4.8 Análisis de delays (intervalos de reaparición)
    gaps_por_numero = defaultdict(list)
    last_idx = {}
    for i, s in enumerate(sorteos):
        for n in s['nums']:
            if n in last_idx:
                gaps_por_numero[n].append(i - last_idx[n])
            last_idx[n] = i

    medias_gap = {n: np.mean(gaps_por_numero[n]) if gaps_por_numero[n] else None
                  for n in range(MAX_NUM + 1)}

    print(f"\n[9] NÚMEROS CON CICLO DE REAPARICIÓN MÁS REGULAR (menor desv. estándar de gaps):")
    regulares = []
    for n in range(MAX_NUM + 1):
        g = gaps_por_numero[n]
        if len(g) >= 5:
            regulares.append((n, np.std(g), np.mean(g), len(g)))
    regulares.sort(key=lambda x: x[1])
    for n, std, mean, cnt in regulares[:10]:
        print(f"  Nº {n:02d}: aparece {cnt} veces | gap medio={mean:.1f} sorteos | std={std:.2f}")

    # 4.9 Patrones por mes/día de semana
    from collections import OrderedDict
    meses = defaultdict(list)
    for s in sorteos:
        mes = int(s['fecha'].split('-')[1])
        meses[mes].extend(s['nums'])

    print(f"\n[10] DISTRIBUCIÓN POR MES (números más frecuentes en cada mes):")
    nombres_mes = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    for mes in sorted(meses.keys()):
        cnt_mes = Counter(meses[mes])
        top3 = [f"{n}({c})" for n,c in cnt_mes.most_common(3)]
        print(f"  {nombres_mes[mes-1]}: {', '.join(top3)}")

    # 4.10 Análisis de decenas
    print(f"\n[11] DISTRIBUCIÓN POR DECENAS:")
    decenas_global = defaultdict(int)
    for s in sorteos:
        for n in s['nums']:
            decenas_global[n // 10] += 1
    total_bolas = len(sorteos) * 6
    for d in sorted(decenas_global.keys()):
        rango = f"{d*10:02d}-{min(d*10+9, 45):02d}"
        cnt = decenas_global[d]
        bar = "#" * int(cnt / total_bolas * 100)
        print(f"  Decena {rango}: {cnt} ({cnt/total_bolas*100:.1f}%) {bar}")

    return freq_global, cooc_global, gaps_por_numero

# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKTESTING WALK-FORWARD DE CADA ALGORITMO
# ─────────────────────────────────────────────────────────────────────────────

def backtest_algoritmo(sorteos, fn_score, nombre, start=30, top_k=15):
    """
    Para cada sorteo t desde `start`:
      1. Usa solo los datos anteriores (sorteos[:t]) como historia.
      2. Puntúa los 46 números con fn_score.
      3. Toma los top_k números como predicción.
      4. Cuenta cuántos coinciden con el sorteo real t.
    """
    aciertos = []
    for t in range(start, len(sorteos)):
        hist = sorteos[:t]
        real = set(sorteos[t]['nums'])
        scores = fn_score(hist)
        ranking = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
        predichos = set(ranking[:top_k])
        hit = len(real.intersection(predichos))
        aciertos.append(hit)

    n_eval = len(aciertos)
    dist = Counter(aciertos)
    media = np.mean(aciertos)
    p3 = sum(1 for h in aciertos if h >= 3) / n_eval * 100
    p4 = sum(1 for h in aciertos if h >= 4) / n_eval * 100
    p6 = sum(1 for h in aciertos if h == 6) / n_eval * 100

    print(f"\n  [{nombre}] top_k={top_k} | evaluados={n_eval}")
    print(f"    Media aciertos/sorteo: {media:.3f}")
    print(f"    3+ aciertos: {p3:.1f}% | 4+: {p4:.1f}% | 6 exacto: {p6:.2f}%")
    dist_str = "  ".join(f"{k}ac={dist[k]}" for k in sorted(dist.keys()))
    print(f"    Distribución: {dist_str}")

    return {'nombre': nombre, 'media': media, 'p3': p3, 'p4': p4,
            'p6': p6, 'dist': dist, 'n_eval': n_eval}

def backtest_aleatorio_control(sorteos, start=30, top_k=15, n_rep=1000):
    """Simulación Monte Carlo de selección aleatoria para comparación."""
    import random
    n_eval = len(sorteos) - start
    total_hits = 0
    hits_3plus = 0
    for t in range(start, len(sorteos)):
        real = set(sorteos[t]['nums'])
        for _ in range(n_rep):
            pred = set(random.sample(range(MAX_NUM + 1), top_k))
            h = len(real.intersection(pred))
            total_hits += h
            if h >= 3:
                hits_3plus += 1
    media = total_hits / (n_eval * n_rep)
    p3 = hits_3plus / (n_eval * n_rep) * 100
    print(f"\n  [CONTROL ALEATORIO] top_k={top_k} (Monte Carlo {n_rep} rep/sorteo)")
    print(f"    Media aciertos/sorteo: {media:.3f}  |  3+ aciertos: {p3:.1f}%")
    return media, p3

# ─────────────────────────────────────────────────────────────────────────────
# 6. ANÁLISIS DE PATRONES ESPECIALES
# ─────────────────────────────────────────────────────────────────────────────

def patron_retorno_post_ausencia(sorteos):
    """
    Hipótesis: un número que lleva X sorteos sin salir tiende a volver.
    Mide la tasa de retorno según el delay actual.
    """
    print("\n[PATRÓN] TASA DE RETORNO SEGÚN DELAY ACTUAL")
    print("  delay | veces que retornó | total casos | tasa(%)")
    print("  ------|-------------------|-------------|--------")

    retornos = defaultdict(lambda: [0, 0])  # delay -> [retornó, total]
    last_seen = {}

    for t, s in enumerate(sorteos):
        nums_t = set(s['nums'])
        # Para cada número que no salió ayer, mide su delay y si sale hoy
        for n in range(MAX_NUM + 1):
            if n in last_seen:
                delay = t - last_seen[n]
                aparecio = 1 if n in nums_t else 0
                retornos[delay][1] += 1
                retornos[delay][0] += aparecio
        # Actualizar last_seen
        for n in nums_t:
            last_seen[n] = t

    # Mostrar solo delays con suficientes muestras
    for delay in sorted(retornos.keys()):
        ret, total = retornos[delay]
        if total >= 10:
            tasa = ret / total * 100
            bar = "#" * int(tasa)
            print(f"  {delay:5d} | {ret:17d} | {total:11d} | {tasa:5.1f}% {bar}")

def patron_numeros_compañeros(sorteos, top_n=10):
    """
    ¿Qué números tienden a salir JUNTOS con más frecuencia de lo esperado?
    Calcula chi-cuadrado de independencia para cada par.
    """
    print(f"\n[PATRÓN] PARES MÁS ASOCIADOS (sobre frecuencia esperada)")
    N = len(sorteos)
    freq = Counter()
    for s in sorteos:
        freq.update(s['nums'])

    cooc = Counter()
    for s in sorteos:
        for a, b in itertools.combinations(s['nums'], 2):
            cooc[(a, b)] += 1

    # Lift = P(a,b) / (P(a)*P(b))
    lifts = []
    for (a, b), cnt_ab in cooc.items():
        p_a = freq[a] / N
        p_b = freq[b] / N
        p_ab = cnt_ab / N
        expected = p_a * p_b
        lift = p_ab / expected if expected > 0 else 0
        lifts.append(((a, b), lift, cnt_ab))

    lifts.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Par':<10} {'Lift':>8} {'Observado':>12} {'Esperado':>12}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*12}")
    for (a, b), lift, cnt_ab in lifts[:top_n]:
        p_a = freq[a] / N
        p_b = freq[b] / N
        expected = p_a * p_b * N
        print(f"  {a:02d}-{b:02d}     {lift:8.3f} {cnt_ab:12d} {expected:12.2f}")

def patron_ciclos_fft(sorteos, top_n=5):
    """
    Detecta ciclos temporales en la aparición de cada número usando FFT.
    """
    print(f"\n[PATRÓN] CICLOS TEMPORALES DOMINANTES (FFT por número)")
    print(f"  Número | Período dominante (sorteos) | Amplitud")
    print(f"  -------|------------------------------|----------")
    N = len(sorteos)
    for n in range(MAX_NUM + 1):
        serie = np.array([1.0 if n in s['nums'] else 0.0 for s in sorteos])
        sc = serie - np.mean(serie)
        fft_v = np.fft.fft(sc)
        freqs = np.fft.fftfreq(N)
        mitad = N // 2
        amps = np.abs(fft_v[:mitad])
        amps[0] = 0
        if len(amps) > 1:
            idx_max = np.argmax(amps)
            freq_dom = freqs[idx_max]
            periodo = 1.0 / freq_dom if freq_dom != 0 else float('inf')
            amp = amps[idx_max]
            if amp > 0.5:  # Solo ciclos significativos
                print(f"  Nº {n:02d}  | {periodo:28.1f} | {amp:.3f}")

def patron_transiciones_markov(sorteos):
    """
    Muestra las transiciones Markov más fuertes: qué número predice mejor al siguiente.
    """
    print(f"\n[PATRÓN] TRANSICIONES MARKOV MÁS FUERTES (P(b|a))")
    trans = matriz_markov(sorteos)
    top_trans = []
    for prev, nexts in trans.items():
        total = sum(nexts.values())
        if total >= 10:
            for nxt, cnt in nexts.items():
                prob = cnt / total
                if prob > 0.25:  # más del 25% de las veces
                    top_trans.append((prev, nxt, prob, cnt, total))

    top_trans.sort(key=lambda x: x[2], reverse=True)
    print(f"  {'Prev->Sig':>10} {'P(sig|prev)':>12} {'Observado':>10} {'Total salidas prev':>20}")
    for prev, nxt, prob, cnt, total in top_trans[:15]:
        bar = "o" * int(prob * 20)
        print(f"  {prev:02d} -> {nxt:02d}   {prob:12.3f} {cnt:10d} {total:20d}  {bar}")

def patron_numeros_calientes_frios(sorteos, ventanas=(10, 20, 50)):
    """
    Clasifica números por su temperatura relativa en distintas ventanas.
    """
    print(f"\n[PATRÓN] TEMPERATURA DE NÚMEROS (frecuencia relativa en distintas ventanas)")
    print(f"  {'Nº':>3} | {'10 ult':>8} | {'20 ult':>8} | {'50 ult':>8} | Tendencia")
    print(f"  {'-'*3}---{'-'*8}---{'-'*8}---{'-'*8}---{'-'*10}")
    for n in range(MAX_NUM + 1):
        freqs = []
        for v in ventanas:
            hist_v = sorteos[-v:] if len(sorteos) >= v else sorteos
            f = sum(1 for s in hist_v if n in s['nums']) / (len(hist_v) * 6) * 6
            freqs.append(f)
        # Tendencia: sube, baja, estable
        if freqs[0] > freqs[2] * 1.3:
            tend = "(+) CALIENTE"
        elif freqs[0] < freqs[2] * 0.7:
            tend = "(-) FRIO"
        else:
            tend = "(=) estable"
        if freqs[0] > 0.2 or freqs[2] > 0.2 or tend != "(=) estable":
            print(f"  {n:02d}  | {freqs[0]:8.3f} | {freqs[1]:8.3f} | {freqs[2]:8.3f} | {tend}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PREDICCIÓN PARA EL PRÓXIMO SORTEO
# ─────────────────────────────────────────────────────────────────────────────

def predecir_proximo(sorteos):
    print("\n" + "="*60)
    print("  PREDICCIÓN PARA EL PRÓXIMO SORTEO")
    print(f"  Basada en todo el histórico hasta: {sorteos[-1]['fecha']}")
    print(f"  Ultimo sorteo: No {sorteos[-1]['sorteo']} -> {sorteos[-1]['nums']}")
    print("="*60)

    scores = score_ensemble(sorteos)
    ranking = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)

    print("\nRanking completo por score del Ensemble:")
    print(f"  {'Nº':>3} | {'Score':>8} | {'Delay':>6} | {'Freq20':>7} | Barra")
    print(f"  {'-'*3}---{'-'*8}---{'-'*6}---{'-'*7}---{'-'*20}")
    del_act = delays_actuales(sorteos)
    freq20 = frecuencias_ventana(sorteos, 20)
    for n in ranking[:20]:
        bar = "#" * int(scores[n] * 1000)
        print(f"  {n:02d}  | {scores[n]:8.5f} | {del_act[n]:6d} | {freq20[n]:7.4f} | {bar}")

    top6 = ranking[:6]
    top12 = ranking[:12]

    print(f"\n  -> BOLETO SUGERIDO (top 6 del ensemble): {sorted(top6)}")
    print(f"  -> POOL EXTENDIDO  (top 12):             {sorted(top12)}")

    # Mostrar scores individuales de cada algoritmo para los top 6
    s_freq   = score_frecuencia(sorteos)
    s_delay  = score_delay_geometrico(sorteos)
    s_markov = score_markov(sorteos)
    s_cooc   = score_coocurrencia(sorteos)
    s_ewma   = score_ewma_crossover(sorteos)
    s_fft    = score_fft_prediccion(sorteos)

    print(f"\n  Detalle por algoritmo para top 6:")
    print(f"  {'Nº':>3} | {'Freq':>7} | {'Delay':>7} | {'Markov':>7} | {'Cooc':>7} | {'EWMA':>7} | {'FFT':>7}")
    for n in sorted(top6):
        print(f"  {n:02d}  | {s_freq.get(n,0):7.4f} | {s_delay.get(n,0):7.4f} | "
              f"{s_markov.get(n,0):7.4f} | {s_cooc.get(n,0):7.4f} | "
              f"{s_ewma.get(n,0):7.4f} | {s_fft.get(n,0):7.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Cargando datos...")
    sorteos = cargar_datos()
    print(f"OK: {len(sorteos)} sorteos cargados ({sorteos[0]['fecha']} -> {sorteos[-1]['fecha']})")

    # ── A. ANÁLISIS ESTADÍSTICO DESCRIPTIVO ──────────────────────────────────
    analisis_estadistico_completo(sorteos)

    # ── B. PATRONES ESPECÍFICOS ───────────────────────────────────────────────
    patron_retorno_post_ausencia(sorteos)
    patron_numeros_compañeros(sorteos, top_n=15)
    patron_transiciones_markov(sorteos)
    patron_ciclos_fft(sorteos)
    patron_numeros_calientes_frios(sorteos)

    # ── C. BACKTESTING WALK-FORWARD DE CADA ALGORITMO ────────────────────────
    print("\n" + "="*60)
    print("  BACKTESTING WALK-FORWARD (predicción vs. realidad histórica)")
    print("  Cada algoritmo sólo usa datos ANTERIORES al sorteo evaluado.")
    print("  top_k=15 significa: se eligen los 15 números con mayor score.")
    print("="*60)

    TOP_K = 15
    START = 50

    resultados = []
    resultados.append(backtest_algoritmo(sorteos, score_frecuencia,         "Frecuencia últimas 20", start=START, top_k=TOP_K))
    resultados.append(backtest_algoritmo(sorteos, score_delay_geometrico,   "Delay Geométrico",      start=START, top_k=TOP_K))
    resultados.append(backtest_algoritmo(sorteos, score_markov,             "Markov Orden 1",        start=START, top_k=TOP_K))
    resultados.append(backtest_algoritmo(sorteos, score_coocurrencia,       "Co-ocurrencia",         start=START, top_k=TOP_K))
    resultados.append(backtest_algoritmo(sorteos, score_ewma_crossover,     "EWMA Crossover",        start=START, top_k=TOP_K))
    resultados.append(backtest_algoritmo(sorteos, score_fft_prediccion,     "FFT Espectral",         start=START, top_k=TOP_K))
    resultados.append(backtest_algoritmo(sorteos, score_ensemble,           "ENSEMBLE (todos)",      start=START, top_k=TOP_K))

    media_azar, p3_azar = backtest_aleatorio_control(sorteos, start=START, top_k=TOP_K, n_rep=200)

    # ── D. TABLA COMPARATIVA FINAL ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TABLA COMPARATIVA: TODOS LOS ALGORITMOS vs. AZAR")
    print("="*60)
    print(f"  {'Algoritmo':<30} {'Media':>7} {'3+%':>8} {'4+%':>8} {'Ventaja':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    for r in resultados:
        ventaja = r['media'] / media_azar if media_azar > 0 else 0
        print(f"  {r['nombre']:<30} {r['media']:7.3f} {r['p3']:8.2f}% {r['p4']:8.2f}% {ventaja:8.3f}x")
    print(f"  {'[CONTROL ALEATORIO]':<30} {media_azar:7.3f} {p3_azar:8.2f}%  {'---':>8}  {'1.000x':>8}")

    # ── E. PREDICCIÓN PRÓXIMO SORTEO ──────────────────────────────────────────
    predecir_proximo(sorteos)

    print("\n[OK] Analisis completado.")

if __name__ == '__main__':
    main()
