"""
Test de Evaluacion de las Nuevas Mejoras:
1. Cobertura Complementaria de Cuadruplas en Boleto 3 (Maximizador de Cuadruplas)
2. Sinergia de Trios Historicos (Afinidades de Orden 3)
3. Combinacion de Ambas
Evaluando 500 sorteos historicos
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

strategies = {
    '0. Produccion Actual (Baseline)': {
        'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'quads_covered': []
    },
    '1. Mejora 1: Cobertura Complementaria Cuadruplas (Boleto 3)': {
        'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'quads_covered': []
    },
    '2. Mejora 2: Sinergia de Trios Historicos (Orden 3)': {
        'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'quads_covered': []
    },
    '3. Combo 1 + 2 (Cuadruplas Complementarias + Sinergia Trios)': {
        'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'quads_covered': []
    }
}

start_time = time.time()

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
    pozo_extra = target_trad | target_seg | target_rev
    
    train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
    
    counts_60 = Counter()
    for d in train_draws[-60:]:
        for n in d['numbers']: counts_60[n] += 1
        
    recency_scores = np.zeros(46)
    for t_idx, d in enumerate(train_draws[-30:]):
        w = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency_scores[n] += w
        
    pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
    ddm_scores = compute_delta_delay_momentum_scores(train_draws)
    attn_scores = compute_fast_attention_mlp_scores(train_draws)
    
    # Precomputar Trios Historicos (ultimos 120 sorteos para no saturar memoria)
    trio_counts = Counter()
    for d in train_draws[-120:]:
        for u, v, w in itertools.combinations(d['numbers'], 3):
            trio_counts[(u, v, w)] += 1
            
    # Regimen
    trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    last_5_trad = trad_series[-5:]
    n_rep_b = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
    avg_rep_b = n_rep_b / 4.0
    if avg_rep_b >= 1.0: w_rec, w_ddm = 2.8, 1.2
    elif avg_rep_b <= 0.5: w_rec, w_ddm = 1.2, 2.8
    else: w_rec, w_ddm = 2.0, 1.8
        
    # Resonancia
    last_sorteo_num = max(d['sorteo'] for d in train_draws)
    last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
    res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
    
    borda_base = defaultdict(float)
    for m, w in [(counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
        rk = sorted(range(46), key=lambda x: m[x], reverse=True)
        for rank, n in enumerate(rk): borda_base[n] += w * (45 - rank)
    for n in range(46):
        if res_counts[n] >= 2: borda_base[n] += 18.0 * (res_counts[n] - 1)
        
    same_day = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day[-40:] for n in d['numbers'])
    borda_day = defaultdict(float)
    borda_hib = defaultdict(float)
    for n in range(46):
        borda_day[n] = borda_base[n] + 16.0 * counts_day[n]
        borda_hib[n] = borda_base[n] + 8.0 * counts_day[n]
        
    saturated_decades = set()
    last_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional'][-1]
    for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
        if cnt >= 3: saturated_decades.add(dec)
    antag = compute_antagonist_pairs(train_draws)
    recent_numbers = set()
    for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
    
    pool_base = select_balanced_pool_9_elite(borda_base, saturated_decades)
    pool_day = select_balanced_pool_9_elite(borda_day, saturated_decades)
    pool_hib = select_balanced_pool_9_elite(borda_hib, saturated_decades)
    
    def get_v(p):
        c_list = list(itertools.combinations(p, 6))
        v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)]
        if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=4)]
        if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
        return v or c_list[:10]
        
    v_base = get_v(pool_base)
    v_day = get_v(pool_day)
    v_hib = get_v(pool_hib)
    
    t_b = select_optimal_wheel_triplet(v_base, pool_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_d = select_optimal_wheel_triplet(v_day, pool_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_h = select_optimal_wheel_triplet(v_hib, pool_hib, borda_hib, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    
    b1 = t_b[0]
    b2 = t_d[0] if t_d[0] != b1 else t_d[1]
    
    # 0. BASELINE ACTUAL
    b3_base = t_h[0] if (t_h[0] != b1 and t_h[0] != b2) else t_b[1]
    t_0 = [b1, b2, b3_base]
    
    # 1. MEJORA 1: COBERTURA COMPLEMENTARIA DE CUADRUPLAS EN B3
    q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
    best_b3_quad = None
    best_score_quad = -float('inf')
    for c in v_hib:
        if c == b1 or c == b2: continue
        q = set(itertools.combinations(c, 4))
        new_q = len(q - q12)
        base_score = sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
        total_s = base_score + 80.0 * new_q # Ponderar ganancia de cuadruplas unicas
        if total_s > best_score_quad:
            best_score_quad = total_s
            best_b3_quad = c
    b3_quad = best_b3_quad or b3_base
    t_1 = [b1, b2, b3_quad]
    
    # 2. MEJORA 2: SINERGIA DE TRIOS HISTORICOS (ORDEN 3)
    def score_with_trios(c, b_dict):
        s_base = sum(b_dict[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
        s_trio = sum(trio_counts[t] for t in itertools.combinations(c, 3))
        return s_base + 45.0 * s_trio
        
    b1_trio = max(v_base, key=lambda c: score_with_trios(c, borda_base))
    b2_trio = max([c for c in v_day if len(set(c) & set(b1_trio)) <= 4] or v_day, key=lambda c: score_with_trios(c, borda_day))
    b3_trio = max([c for c in v_hib if c != b1_trio and c != b2_trio] or v_hib, key=lambda c: score_with_trios(c, borda_hib))
    t_2 = [b1_trio, b2_trio, b3_trio]
    
    # 3. COMBO 1 + 2
    q12_trio = set(itertools.combinations(b1_trio, 4)) | set(itertools.combinations(b2_trio, 4))
    best_b3_combo = None
    best_s_combo = -float('inf')
    for c in v_hib:
        if c == b1_trio or c == b2_trio: continue
        q = set(itertools.combinations(c, 4))
        new_q = len(q - q12_trio)
        s_val = score_with_trios(c, borda_hib) + 80.0 * new_q
        if s_val > best_s_combo:
            best_s_combo = s_val
            best_b3_combo = c
    b3_combo = best_b3_combo or b3_trio
    t_3 = [b1_trio, b2_trio, b3_combo]
    
    # Evaluacion
    for name, t_list in [('0. Produccion Actual (Baseline)', t_0),
                         ('1. Mejora 1: Cobertura Complementaria Cuadruplas (Boleto 3)', t_1),
                         ('2. Mejora 2: Sinergia de Trios Historicos (Orden 3)', t_2),
                         ('3. Combo 1 + 2 (Cuadruplas Complementarias + Sinergia Trios)', t_3)]:
        hits_trad = [len(set(t) & target_trad) for t in t_list]
        max_h_trad = max(hits_trad)
        max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
        max_pe = max(len(set(t) & pozo_extra) for t in t_list)
        
        q_cov = len(set(itertools.combinations(t_list[0], 4)) | set(itertools.combinations(t_list[1], 4)) | set(itertools.combinations(t_list[2], 4)))
        strategies[name]['quads_covered'].append(q_cov)
        
        strategies[name]['max_trad'].append(max_h_trad)
        if max_h_trad >= 3: strategies[name]['trad_3+'] += 1
        if max_h_trad >= 4: strategies[name]['trad_4+'] += 1
        if max_any >= 4: strategies[name]['any_4+'] += 1
        if max_pe >= 5: strategies[name]['pe_5+'] += 1
        
    if (step_i + 1) % 100 == 0:
        print(f" -> Progreso: {step_i + 1} de {n_eval} sorteos completados...")

elapsed = time.time() - start_time
print(f"\n[OK] Benchmark completado en {elapsed:.1f} segundos.\n")

print("="*125)
print(f"{'Estrategia Evaluada (500 Sorteos)':<62} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
print("="*125)
for name, s in strategies.items():
    avg_t = np.mean(s['max_trad'])
    p3 = (s['trad_3+'] / n_eval) * 100
    n4_t = s['trad_4+']
    p4_t = (n4_t / n_eval) * 100
    n4_any = s['any_4+']
    p4_any = (n4_any / n_eval) * 100
    pe5 = s['pe_5+']
    avg_quads = np.mean(s['quads_covered'])
    print(f"{name:<62} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}] (Cuadruplas: {avg_quads:.1f}/45)")
print("="*125)

