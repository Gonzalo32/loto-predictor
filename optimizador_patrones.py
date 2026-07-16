"""
=============================================================================
  OPTIMIZADOR DE PATRONES - LOTO
  Mejoras sobre analizador_patrones.py:
    1. Grid search sobre top_k
    2. Ensemble adaptivo (pesos por rendimiento reciente)
    3. Markov de orden 2
    4. Rank fusion (Borda count)
    5. Score de "momentum"
=============================================================================
"""

import csv
import os
import sys
import math
import itertools
import numpy as np
from collections import defaultdict, Counter

MAX_NUM = 45

def cargar_datos(path=r'c:\Users\Administrador\Desktop\lot\historico_loto.csv'):
    sorteos = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nums = sorted([int(row[f'B{i}']) for i in range(1, 7)])
                sorteos.append({'sorteo': int(row['Sorteo']), 'fecha': row['Fecha'], 'nums': nums})
            except Exception:
                continue
    return sorteos

def normalizar(d):
    total = sum(d.values()) or 1.0
    return {k: v / total for k, v in d.items()}

def score_frecuencia(hist, ventana=20):
    freq = {n: 0.0 for n in range(MAX_NUM + 1)}
    rec = hist[-ventana:] if len(hist) >= ventana else hist
    total = len(rec) * 6 or 1
    for s in rec:
        for n in s['nums']:
            freq[n] += 1.0
    return {n: freq[n] / total for n in freq}

def score_delay_geo(hist):
    gaps = defaultdict(list)
    last = {}
    for i, s in enumerate(hist):
        for n in s['nums']:
            if n in last:
                gaps[n].append(i - last[n])
            last[n] = i
    n_total = len(hist)
    score = {}
    for n in range(MAX_NUM + 1):
        g = gaps.get(n, [])
        mu = float(np.mean(g)) if g else 7.6
        delay = n_total - 1 - last.get(n, n_total - 1) if n in last else n_total
        p = (1.0 / mu) * ((1 - 1.0 / mu) ** max(delay - 1, 0)) if mu > 0 else 0.0
        score[n] = p
    return score

def score_markov1(hist):
    if len(hist) < 2:
        return {n: 1/46 for n in range(MAX_NUM + 1)}
    trans = defaultdict(lambda: defaultdict(int))
    for i in range(1, len(hist)):
        for p in hist[i-1]['nums']:
            for c in hist[i]['nums']:
                trans[p][c] += 1
    ultimo = hist[-1]['nums']
    score = defaultdict(float)
    for p in ultimo:
        total = sum(trans[p].values())
        if total > 0:
            for n, cnt in trans[p].items():
                score[n] += cnt / total
    return normalizar(dict(score))

def score_markov2(hist):
    if len(hist) < 3:
        return score_markov1(hist)
    trans2 = defaultdict(lambda: defaultdict(int))
    for i in range(2, len(hist)):
        estado = (frozenset(hist[i-2]['nums']), frozenset(hist[i-1]['nums']))
        for c in hist[i]['nums']:
            trans2[estado][c] += 1
    estado_actual = (frozenset(hist[-2]['nums']), frozenset(hist[-1]['nums']))
    score = defaultdict(float)
    total_t = sum(trans2[estado_actual].values())
    if total_t > 0:
        for n, cnt in trans2[estado_actual].items():
            score[n] = cnt / total_t
    if sum(score.values()) == 0:
        return score_markov1(hist)
    return normalizar(dict(score))

def score_coocurrencia(hist):
    if len(hist) < 2:
        return {n: 1/46 for n in range(MAX_NUM + 1)}
    cooc = defaultdict(lambda: defaultdict(int))
    for s in hist:
        for a, b in itertools.combinations(s['nums'], 2):
            cooc[a][b] += 1
            cooc[b][a] += 1
    ultimo = set(hist[-1]['nums'])
    score = defaultdict(float)
    for n in range(MAX_NUM + 1):
        if n not in ultimo:
            score[n] = sum(cooc[n].get(u, 0) for u in ultimo)
    return normalizar(dict(score))

def score_ewma(hist):
    score = {}
    for n in range(MAX_NUM + 1):
        serie = np.array([1.0 if n in s['nums'] else 0.0 for s in hist])
        if len(serie) < 20:
            score[n] = float(np.mean(serie)) if len(serie) > 0 else 0.0
            continue
        def _e(data, span):
            a = 2.0 / (span + 1)
            v = data[0]
            for x in data[1:]: v = a * x + (1 - a) * v
            return v
        v3, v10, v30 = _e(serie, 3), _e(serie, 10), _e(serie, 30)
        score[n] = 0.5 * v10 + 0.5 * (1.0 if v3 > v30 else 0.0)
    return score

def score_fft(hist):
    n_t = len(hist)
    score = {}
    for n in range(MAX_NUM + 1):
        serie = np.array([1.0 if n in s['nums'] else 0.0 for s in hist])
        if n_t < 8:
            score[n] = float(np.mean(serie)) if n_t > 0 else 0.0
            continue
        sc = serie - np.mean(serie)
        fft_v = np.fft.fft(sc)
        freqs = np.fft.fftfreq(n_t)
        mitad = n_t // 2
        amps = np.abs(fft_v[:mitad])
        amps[0] = 0
        top = np.argsort(amps)[-3:]
        proy = sum(amps[i] * np.cos(2*np.pi*freqs[i]*n_t + np.angle(fft_v[i])) for i in top if amps[i] > 0)
        score[n] = float(np.mean(serie) + proy / n_t)
    return score

def score_momentum(hist, ventanas=(5, 10, 20)):
    score = {}
    for n in range(MAX_NUM + 1):
        freqs = []
        for v in ventanas:
            bloque = hist[-v:] if len(hist) >= v else hist
            f = sum(1 for s in bloque if n in s['nums']) / (len(bloque) or 1)
            freqs.append(f)
        ratio = freqs[0] / (freqs[-1] + 1e-6)
        score[n] = ratio * freqs[0]
    return score

ALGORITMOS = {
    'freq':     score_frecuencia,
    'delay':    score_delay_geo,
    'markov1':  score_markov1,
    'markov2':  score_markov2,
    'cooc':     score_coocurrencia,
    'ewma':     score_ewma,
    'fft':      score_fft,
    'momentum': score_momentum,
}

def rank_fusion(score_dicts, pesos=None):
    numeros = list(range(MAX_NUM + 1))
    if pesos is None:
        pesos = [1.0] * len(score_dicts)
    borda = defaultdict(float)
    for scores, peso in zip(score_dicts, pesos):
        ranking = sorted(numeros, key=lambda n: scores.get(n, 0), reverse=True)
        for rango, n in enumerate(ranking):
            borda[n] += (MAX_NUM - rango) * peso
    return dict(borda)

def actualizar_pesos(acc_por_algo, ventana=50):
    nuevos = {}
    for nombre in ALGORITMOS.keys():
        rec = acc_por_algo[nombre][-ventana:]
        nuevos[nombre] = float(np.mean(rec)) if rec else 1.0
    vals = np.array([nuevos[k] for k in ALGORITMOS.keys()])
    vals = vals - vals.min() + 0.1
    std = vals.std()
    exp_v = np.exp(vals / std if std > 0 else vals)
    exp_v /= exp_v.sum()
    return {k: float(exp_v[i]) for i, k in enumerate(ALGORITMOS.keys())}

def backtest_simple(sorteos, fn_score, nombre, start=50, top_k=15):
    aciertos = []
    for t in range(start, len(sorteos)):
        real = set(sorteos[t]['nums'])
        sc = fn_score(sorteos[:t])
        ranking = sorted(sc.keys(), key=lambda n: sc.get(n, 0), reverse=True)
        aciertos.append(len(real & set(ranking[:top_k])))
    n = len(aciertos)
    dist = Counter(aciertos)
    media = np.mean(aciertos)
    p3 = sum(1 for h in aciertos if h >= 3) / n * 100
    p4 = sum(1 for h in aciertos if h >= 4) / n * 100
    print(f"  [{nombre}] media={media:.3f} | 3+={p3:.1f}% | 4+={p4:.1f}%")
    print(f"    Dist: { '  '.join(f'{k}ac={dist[k]}' for k in sorted(dist.keys())) }")
    return {'nombre': nombre, 'media': media, 'p3': p3, 'p4': p4, 'aciertos': aciertos}

def gridsearch_topk(sorteos, start=50, ks=(10, 12, 14, 15, 16, 18, 20, 22)):
    print("\n" + "="*65)
    print("  GRID SEARCH: top_k optimo para Ensemble con Rank Fusion")
    print("="*65)
    print(f"  {'top_k':>6} | {'Media':>8} | {'3+%':>8} | {'4+%':>8} | Score")
    print(f"  {'-'*6}---{'-'*8}---{'-'*8}---{'-'*8}---{'-'*8}")
    mejores = []
    for k in ks:
        aciertos = []
        for t in range(start, len(sorteos)):
            real = set(sorteos[t]['nums'])
            hist = sorteos[:t]
            sc_list = [fn(hist) for fn in ALGORITMOS.values()]
            borda = rank_fusion(sc_list)
            ranking = sorted(borda.keys(), key=lambda n: borda.get(n, 0), reverse=True)
            aciertos.append(len(real & set(ranking[:k])))
        n = len(aciertos)
        media = float(np.mean(aciertos))
        p3 = sum(1 for h in aciertos if h >= 3) / n * 100
        p4 = sum(1 for h in aciertos if h >= 4) / n * 100
        sc = media + 0.02 * p3 + 0.04 * p4
        mejores.append((k, media, p3, p4, sc))
        print(f"  {k:>6} | {media:8.3f} | {p3:8.2f}% | {p4:8.2f}% | {sc:.4f}")
    mejor = max(mejores, key=lambda x: x[4])
    print(f"\n  >>> MEJOR top_k={mejor[0]}  (media={mejor[1]:.3f}, 3+={mejor[2]:.2f}%, 4+={mejor[3]:.2f}%)")
    return mejor[0]

def backtest_adaptivo(sorteos, start=50, top_k=15, ventana=50):
    pesos = {k: 1.0 / len(ALGORITMOS) for k in ALGORITMOS}
    acc_por_algo = {k: [] for k in ALGORITMOS}
    aciertos_ens = []
    for t in range(start, len(sorteos)):
        hist = sorteos[:t]
        real = set(sorteos[t]['nums'])
        sc_ind = {}
        for nombre, fn in ALGORITMOS.items():
            sc = fn(hist)
            ranking = sorted(sc.keys(), key=lambda n: sc.get(n, 0), reverse=True)
            hit = len(real & set(ranking[:top_k]))
            acc_por_algo[nombre].append(hit)
            sc_ind[nombre] = sc
        sc_list = [sc_ind[k] for k in ALGORITMOS]
        pesos_list = [pesos[k] for k in ALGORITMOS]
        borda = rank_fusion(sc_list, pesos=pesos_list)
        ranking_ens = sorted(borda.keys(), key=lambda n: borda.get(n, 0), reverse=True)
        aciertos_ens.append(len(real & set(ranking_ens[:top_k])))
        if t >= start + ventana:
            pesos = actualizar_pesos(acc_por_algo, ventana)
        if (t - start + 1) % 100 == 0:
            pct = (t - start + 1) / (len(sorteos) - start) * 100
            print(f"    {pct:.0f}% | pesos: { {k: round(v,3) for k,v in pesos.items()} }")
    n = len(aciertos_ens)
    media = float(np.mean(aciertos_ens))
    p3 = sum(1 for h in aciertos_ens if h >= 3) / n * 100
    p4 = sum(1 for h in aciertos_ens if h >= 4) / n * 100
    dist = Counter(aciertos_ens)
    print(f"\n  [ENSEMBLE ADAPTIVO] media={media:.3f} | 3+={p3:.1f}% | 4+={p4:.1f}%")
    print(f"  Dist: { '  '.join(f'{k}ac={dist[k]}' for k in sorted(dist.keys())) }")
    print(f"  Pesos finales: { {k: round(v, 4) for k,v in pesos.items()} }")
    return {'media': media, 'p3': p3, 'p4': p4, 'pesos_finales': pesos, 'aciertos': aciertos_ens}

def control_aleatorio(sorteos, start=50, top_k=15, n_rep=300):
    import random
    total_h, h3 = 0, 0
    n_eval = len(sorteos) - start
    for t in range(start, len(sorteos)):
        real = set(sorteos[t]['nums'])
        for _ in range(n_rep):
            pred = set(random.sample(range(MAX_NUM + 1), top_k))
            h = len(real & pred)
            total_h += h
            if h >= 3: h3 += 1
    media = total_h / (n_eval * n_rep)
    p3 = h3 / (n_eval * n_rep) * 100
    print(f"\n  [AZAR PURO] top_k={top_k} | media={media:.4f} | 3+={p3:.2f}%")
    return media, p3

def predecir(sorteos, pesos_finales, top_k=15):
    print("\n" + "="*65)
    print("  PREDICCION PROXIMO SORTEO")
    print(f"  Ultimo: {sorteos[-1]['sorteo']} ({sorteos[-1]['fecha']}) -> {sorteos[-1]['nums']}")
    print("="*65)
    sc_ind = {nombre: fn(sorteos) for nombre, fn in ALGORITMOS.items()}
    sc_list = [sc_ind[k] for k in ALGORITMOS]
    pesos_list = [pesos_finales.get(k, 1.0) for k in ALGORITMOS]
    borda = rank_fusion(sc_list, pesos=pesos_list)
    ranking = sorted(borda.keys(), key=lambda n: borda.get(n, 0), reverse=True)
    print(f"\n  {'No':>3} | {'Borda':>8} | {'Freq':>7} | {'Markov1':>8} | {'Markov2':>8} | {'Momentum':>9}")
    print(f"  {'-'*3}---{'-'*8}---{'-'*7}---{'-'*8}---{'-'*8}---{'-'*9}")
    for n in ranking[:top_k + 5]:
        print(f"  {n:02d}  | {borda[n]:8.1f} | {sc_ind['freq'].get(n,0):7.4f} | "
              f"{sc_ind['markov1'].get(n,0):8.4f} | {sc_ind['markov2'].get(n,0):8.4f} | "
              f"{sc_ind['momentum'].get(n,0):9.5f}")
    top6 = sorted(ranking[:6])
    topk = sorted(ranking[:top_k])
    print(f"\n  >>> BOLETO SUGERIDO (top 6): {top6}")
    print(f"  >>> POOL EXTENDIDO  (top {top_k}): {topk}")
    return top6, topk

def tabla_final(resultados_base, resultado_adaptivo, media_azar, p3_azar, top_k):
    print("\n" + "="*65)
    print(f"  TABLA COMPARATIVA FINAL (top_k={top_k})")
    print("="*65)
    print(f"  {'Metodo':<28} {'Media':>7} {'3+%':>8} {'4+%':>8} {'Ventaja':>9}")
    print(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")
    for r in resultados_base:
        v = r['media'] / media_azar if media_azar > 0 else 0
        print(f"  {r['nombre']:<28} {r['media']:7.3f} {r['p3']:8.2f}% {r['p4']:8.2f}% {v:9.3f}x")
    v_a = resultado_adaptivo['media'] / media_azar if media_azar > 0 else 0
    print(f"  {'ENSEMBLE ADAPTIVO (NUEVO)':<28} {resultado_adaptivo['media']:7.3f} "
          f"{resultado_adaptivo['p3']:8.2f}% {resultado_adaptivo['p4']:8.2f}% {v_a:9.3f}x  <-- NUEVO")
    print(f"  {'[AZAR PURO]':<28} {media_azar:7.3f} {p3_azar:8.2f}%      ---     1.000x")
    print("="*65)
    mejor_base = max(r['media'] for r in resultados_base)
    print(f"\n  Mejora sobre mejor metodo anterior (media): {(resultado_adaptivo['media']-mejor_base)/mejor_base*100:+.2f}%")
    print(f"  Mejora sobre azar puro (media):             {(resultado_adaptivo['media']/media_azar-1)*100:+.2f}%")
    print(f"  Mejora en 3+ aciertos vs anterior:         {resultado_adaptivo['p3'] - max(r['p3'] for r in resultados_base):+.2f} pp")

def main():
    print("Cargando datos...")
    sorteos = cargar_datos()
    print(f"OK: {len(sorteos)} sorteos ({sorteos[0]['fecha']} -> {sorteos[-1]['fecha']})")
    START = 50

    top_k = gridsearch_topk(sorteos, start=START)

    print(f"\n{'='*65}")
    print(f"  BACKTEST INDIVIDUAL (top_k={top_k})")
    print(f"{'='*65}")
    resultados_base = [backtest_simple(sorteos, fn, nombre, start=START, top_k=top_k)
                       for nombre, fn in ALGORITMOS.items()]

    print("\n  Calculando control aleatorio...")
    media_azar, p3_azar = control_aleatorio(sorteos, start=START, top_k=top_k, n_rep=300)

    print(f"\n{'='*65}")
    print(f"  ENSEMBLE ADAPTIVO (top_k={top_k})")
    print(f"{'='*65}")
    resultado_adaptivo = backtest_adaptivo(sorteos, start=START, top_k=top_k)

    tabla_final(resultados_base, resultado_adaptivo, media_azar, p3_azar, top_k)

    predecir(sorteos, resultado_adaptivo['pesos_finales'], top_k=top_k)
    print("\nLISTO.")

if __name__ == '__main__':
    main()
