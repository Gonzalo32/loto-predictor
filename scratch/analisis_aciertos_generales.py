"""
Analisis Detallado de Aciertos Generales:
Comparacion de metricas globales de aciertos en 500 sorteos (#2924 a #3423):
1. Azar puro (3 boletos aleatorios validos)
2. Algoritmo Viejo (Pool 12 disperso)
3. Algoritmo Nuevo en Produccion (Portfolio Especializado + Resonancia + BIBD)
"""
import os, sys, itertools, time
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_antagonist_pairs,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores
)
from cobertura_combinatoria import select_optimal_wheel_triplet
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores
from scratch.test_nuevas_mejoras_resonancia import select_balanced_pool_9_elite

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
n_eval = 500
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]

stats = {
    'Azar Puro (Aleatorio)': {
        'hist_trad': Counter(), 'hist_any': Counter(), 'total_hits_trad': 0, 'total_hits_all_mods': 0
    },
    'Algoritmo Viejo (Pool 12 Disperso)': {
        'hist_trad': Counter(), 'hist_any': Counter(), 'total_hits_trad': 0, 'total_hits_all_mods': 0
    },
    'Algoritmo Nuevo (Portfolio Especializado)': {
        'hist_trad': Counter(), 'hist_any': Counter(), 'total_hits_trad': 0, 'total_hits_all_mods': 0
    }
}

np.random.seed(42)

for step_i, target_draw in enumerate(eval_draws):
    target_sorteo = target_draw['sorteo']
    target_fecha = target_draw['fecha']
    target_weekday = datetime.strptime(target_fecha, '%Y-%m-%d').weekday()
    
    group = [d for d in draws if d['sorteo'] == target_sorteo]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in group}
    target_trad = real_by_mod.get('Tradicional', set())
    target_seg = real_by_mod.get('Segunda', set())
    target_rev = real_by_mod.get('Revancha', set())
    target_ss = real_by_mod.get('SiempreSale', set())
    all_mods = [target_trad, target_seg, target_rev, target_ss]
    
    train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
    
    # 1. Azar puro: 3 boletos validos al azar
    all_combos_sample = []
    while len(all_combos_sample) < 3:
        rand_t = tuple(sorted(np.random.choice(46, 6, replace=False)))
        if es_ticket_valido(rand_t, min_distinct_decades=3):
            all_combos_sample.append(rand_t)
            
    # 2. Algoritmo Viejo
    counts_60 = Counter(n for d in train_draws[-60:] for n in d['numbers'])
    recency = [0]*46
    for t_idx, d in enumerate(train_draws[-30:]):
        w = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency[n] += w
    pcr, co_mat = compute_exponential_cooccurrence_scores(train_draws)
    ddm = compute_delta_delay_momentum_scores(train_draws)
    attn = compute_fast_attention_mlp_scores(train_draws)
    
    borda_old = defaultdict(float)
    for m, w in [(counts_60, 1.5), (recency, 2.0), (pcr, 1.4), (ddm, 1.8), ({n: attn[n] for n in range(46)}, 1.5)]:
        rk = sorted(range(46), key=lambda x: m[x], reverse=True)
        for rank, n in enumerate(rk): borda_old[n] += w * (45 - rank)
    rk_old = sorted(range(46), key=lambda x: borda_old[x], reverse=True)
    pool_12 = sorted(rk_old[:12])
    c_12 = list(itertools.combinations(pool_12, 6))
    v_12 = [c for c in c_12 if es_ticket_valido(c, min_distinct_decades=3)] or c_12[:10]
    t_old = select_optimal_wheel_triplet(v_12, pool_12, borda_old, co_mat, max_overlap=3, w_cooccur=200.0)
    
    # 3. Algoritmo Nuevo
    trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    last_5_trad = trad_series[-5:]
    n_rep_b = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
    avg_rep_b = n_rep_b / 4.0
    if avg_rep_b >= 1.0: w_rec_b, w_ddm_b = 2.8, 1.2
    elif avg_rep_b <= 0.5: w_rec_b, w_ddm_b = 1.2, 2.8
    else: w_rec_b, w_ddm_b = 2.0, 1.8
        
    last_sorteo_num = max(d['sorteo'] for d in train_draws)
    last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
    res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
    
    borda_new = defaultdict(float)
    for m, w in [(counts_60, 1.5), (recency, w_rec_b), (pcr, 1.4), (ddm, w_ddm_b), ({n: attn[n] for n in range(46)}, 1.5)]:
        rk = sorted(range(46), key=lambda x: m[x], reverse=True)
        for rank, n in enumerate(rk): borda_new[n] += w * (45 - rank)
    for n in range(46):
        if res_counts[n] >= 2: borda_new[n] += 18.0 * (res_counts[n] - 1)
        
    same_day = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day[-40:] for n in d['numbers'])
    borda_day_new = defaultdict(float)
    borda_hib_new = defaultdict(float)
    for n in range(46):
        borda_day_new[n] = borda_new[n] + 16.0 * counts_day[n]
        borda_hib_new[n] = borda_new[n] + 8.0 * counts_day[n]
        
    saturated_decades = set()
    last_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional'][-1]
    for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
        if cnt >= 3: saturated_decades.add(dec)
    antag = compute_antagonist_pairs(train_draws)
    recent_numbers = set()
    for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
    
    pool_base = select_balanced_pool_9_elite(borda_new, saturated_decades)
    pool_day = select_balanced_pool_9_elite(borda_day_new, saturated_decades)
    pool_hib = select_balanced_pool_9_elite(borda_hib_new, saturated_decades)
    
    def get_v(p):
        c_list = list(itertools.combinations(p, 6))
        v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)]
        if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
        return v or c_list[:10]
        
    t_b = select_optimal_wheel_triplet(get_v(pool_base), pool_base, borda_new, co_mat, max_overlap=4, w_cooccur=350.0)
    t_d = select_optimal_wheel_triplet(get_v(pool_day), pool_day, borda_day_new, co_mat, max_overlap=4, w_cooccur=350.0)
    t_h = select_optimal_wheel_triplet(get_v(pool_hib), pool_hib, borda_hib_new, co_mat, max_overlap=4, w_cooccur=350.0)
    
    b1 = t_b[0]
    b2 = t_d[0] if t_d[0] != b1 else t_d[1]
    b3 = t_h[0] if (t_h[0] != b1 and t_h[0] != b2) else t_b[1]
    t_new = [b1, b2, b3]
    
    # Evaluar los 3 modelos
    candidates = [
        ('Azar Puro (Aleatorio)', all_combos_sample),
        ('Algoritmo Viejo (Pool 12 Disperso)', t_old),
        ('Algoritmo Nuevo (Portfolio Especializado)', t_new)
    ]
    
    for name, t_list in candidates:
        hits_t = [len(set(t) & target_trad) for t in t_list]
        max_h_t = max(hits_t)
        max_any = max(max(len(set(t) & m) for t in t_list) for m in all_mods if m)
        
        stats[name]['hist_trad'][max_h_t] += 1
        stats[name]['hist_any'][max_any] += 1
        stats[name]['total_hits_trad'] += sum(hits_t)
        stats[name]['total_hits_all_mods'] += sum(sum(len(set(t) & m) for t in t_list) for m in all_mods if m)

print("="*95)
print(" COMPARATIVA DE ACIERTOS GENERALES EN 500 SORTEOS (1.500 BOLETOS JUGADOS)")
print("="*95)

print(f"\n1. DISTRIBUCION DEL MAXIMO DE ACIERTOS POR SORTEO EN TRADICIONAL:")
print(f"{'Modelo':<42} | {'0 Aciertos':<11} | {'1 Acierto':<11} | {'2 Aciertos':<11} | {'3 Aciertos':<11} | {'4+ Aciertos'}")
print("-" * 105)
for name, d in stats.items():
    h = d['hist_trad']
    p0 = h[0] / n_eval * 100
    p1 = h[1] / n_eval * 100
    p2 = h[2] / n_eval * 100
    p3 = h[3] / n_eval * 100
    p4 = sum(h[k] for k in range(4, 7)) / n_eval * 100
    n4 = sum(h[k] for k in range(4, 7))
    print(f"{name:<42} | {h[0]} ({p0:.1f}%) | {h[1]} ({p1:.1f}%) | {h[2]} ({p2:.1f}%) | {h[3]} ({p3:.1f}%) | {n4} ({p4:.1f}%)")

print(f"\n2. DISTRIBUCION DEL MAXIMO DE ACIERTOS EN CUALQUIER MODALIDAD:")
print(f"{'Modelo':<42} | {'<=1 Acierto':<12} | {'2 Aciertos':<12} | {'3 Aciertos':<12} | {'4+ Aciertos (Premios)'}")
print("-" * 95)
for name, d in stats.items():
    h = d['hist_any']
    p_le1 = (h[0] + h[1]) / n_eval * 100
    p2 = h[2] / n_eval * 100
    p3 = h[3] / n_eval * 100
    n4 = sum(h[k] for k in range(4, 7))
    p4 = n4 / n_eval * 100
    print(f"{name:<42} | {h[0]+h[1]} ({p_le1:.1f}%)  | {h[2]} ({p2:.1f}%)  | {h[3]} ({p3:.1f}%)  | {n4} ({p4:.1f}%)")

print(f"\n3. TOTALES ACUMULADOS DE ACIERTOS (BOLILLAS COINCIDENTES EN CASILLEROS):")
print(f"{'Modelo':<42} | {'Total Aciertos Trad.':<22} | {'Total Aciertos 4 Modalidades'}")
print("-" * 85)
for name, d in stats.items():
    print(f"{name:<42} | {d['total_hits_trad']:<22} | {d['total_hits_all_mods']}")
print("="*95)

