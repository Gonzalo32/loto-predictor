"""
Test de Especializacion Pura por Modalidad:
Boleto 1: Especialista en Tradicional
Boleto 2: Especialista en La Segunda (entrenado con serie de Segunda)
Boleto 3: Especialista en Siempre Sale (entrenado con serie de Siempre Sale)
vs
Baseline de Produccion Actual (B1 Trad + B2 Dia + B3 Consenso)
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
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    select_balanced_pool_9
)
from cobertura_combinatoria import select_optimal_wheel_triplet
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores
from scratch.test_nuevas_mejoras_resonancia import select_balanced_pool_9_elite

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
n_eval = 500
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]

results = {
    'Baseline Produccion Actual': {
        'trad_4+': 0, 'seg_4+': 0, 'ss_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'trad_3+': 0
    },
    'Especialistas por Modalidad (B1 Trad + B2 Segunda + B3 SiempreSale)': {
        'trad_4+': 0, 'seg_4+': 0, 'ss_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'trad_3+': 0
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
    
    # -------------------------------------------------------------
    # 1. BASELINE ACTUAL
    # -------------------------------------------------------------
    counts_60 = Counter(n for d in train_draws[-60:] for n in d['numbers'])
    recency = np.zeros(46)
    for t_idx, d in enumerate(train_draws[-30:]):
        w = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency[n] += w
    pcr, co_mat = compute_exponential_cooccurrence_scores(train_draws)
    ddm = compute_delta_delay_momentum_scores(train_draws)
    attn = compute_fast_attention_mlp_scores(train_draws)
    
    trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    last_5_trad = trad_series[-5:]
    n_rep_b = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
    avg_rep_b = n_rep_b / 4.0
    if avg_rep_b >= 1.0: w_rec, w_ddm = 2.8, 1.2
    elif avg_rep_b <= 0.5: w_rec, w_ddm = 1.2, 2.8
    else: w_rec, w_ddm = 2.0, 1.8
        
    last_sorteo_num = max(d['sorteo'] for d in train_draws)
    last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
    res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
    
    borda_base = defaultdict(float)
    for m, w in [(counts_60, 1.5), (recency, w_rec), (pcr, 1.4), (ddm, w_ddm), ({n: attn[n] for n in range(46)}, 1.5)]:
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
    recent_numbers = set(n for d in train_draws[-5:] for n in d['numbers'])
    
    pool_base = select_balanced_pool_9_elite(borda_base, saturated_decades)
    pool_day = select_balanced_pool_9_elite(borda_day, saturated_decades)
    pool_hib = select_balanced_pool_9_elite(borda_hib, saturated_decades)
    
    def get_v(p):
        c_list = list(itertools.combinations(p, 6))
        v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)]
        if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=4)]
        if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
        return v or c_list[:10]
        
    t_b = select_optimal_wheel_triplet(get_v(pool_base), pool_base, borda_base, co_mat, max_overlap=4, w_cooccur=350.0)
    t_d = select_optimal_wheel_triplet(get_v(pool_day), pool_day, borda_day, co_mat, max_overlap=4, w_cooccur=350.0)
    t_h = select_optimal_wheel_triplet(get_v(pool_hib), pool_hib, borda_hib, co_mat, max_overlap=4, w_cooccur=350.0)
    
    b1_act = t_b[0]
    b2_act = t_d[0] if t_d[0] != b1_act else t_d[1]
    b3_act = t_h[0] if (t_h[0] != b1_act and t_h[0] != b2_act) else t_b[1]
    t_actual = [b1_act, b2_act, b3_act]
    
    # -------------------------------------------------------------
    # 2. MODELO ESPECIALISTAS POR MODALIDAD
    # Boleto 1: Borda Tradicional
    # Boleto 2: Borda entrenado con Segunda
    # Boleto 3: Borda entrenado con SiempreSale
    # -------------------------------------------------------------
    seg_draws = [d for d in train_draws if d['modalidad'] == 'Segunda']
    counts_seg = Counter(n for d in seg_draws[-40:] for n in d['numbers'])
    borda_seg = defaultdict(float)
    for n in range(46):
        borda_seg[n] = borda_base[n] + 16.0 * counts_seg[n]
    pool_seg = select_balanced_pool_9_elite(borda_seg, saturated_decades)
    t_seg = select_optimal_wheel_triplet(get_v(pool_seg), pool_seg, borda_seg, co_mat, max_overlap=4, w_cooccur=350.0)
    
    ss_draws = [d for d in train_draws if d['modalidad'] == 'SiempreSale']
    counts_ss = Counter(n for d in ss_draws[-40:] for n in d['numbers'])
    borda_ss = defaultdict(float)
    for n in range(46):
        borda_ss[n] = borda_base[n] + 16.0 * counts_ss[n]
    pool_ss = select_balanced_pool_9_elite(borda_ss, saturated_decades)
    t_ss = select_optimal_wheel_triplet(get_v(pool_ss), pool_ss, borda_ss, co_mat, max_overlap=4, w_cooccur=350.0)
    
    b1_mod = t_b[0]
    b2_mod = t_seg[0] if t_seg[0] != b1_mod else t_seg[1]
    b3_mod = t_ss[0] if (t_ss[0] != b1_mod and t_ss[0] != b2_mod) else t_b[1]
    t_mod = [b1_mod, b2_mod, b3_mod]
    
    # Evaluacion de ambos
    for name, t_list in [('Baseline Produccion Actual', t_actual),
                         ('Especialistas por Modalidad (B1 Trad + B2 Segunda + B3 SiempreSale)', t_mod)]:
        h_trad = [len(set(t) & target_trad) for t in t_list]
        h_seg = [len(set(t) & target_seg) for t in t_list] if target_seg else [0]
        h_ss = [len(set(t) & target_ss) for t in t_list] if target_ss else [0]
        max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
        max_pe = max(len(set(t) & pozo_extra) for t in t_list)
        
        if max(h_trad) >= 3: results[name]['trad_3+'] += 1
        if max(h_trad) >= 4: results[name]['trad_4+'] += 1
        if max(h_seg) >= 4: results[name]['seg_4+'] += 1
        if max(h_ss) >= 4: results[name]['ss_4+'] += 1
        if max_any >= 4: results[name]['any_4+'] += 1
        if max_pe >= 5: results[name]['pe_5+'] += 1

print("\n" + "="*115)
print(f"{'Estrategia Evaluada (500 Sorteos)':<65} | {'4+ Trad':<10} | {'4+ Segunda':<12} | {'4+ SiempreSale':<15} | {'4+ Cualquier Mod'}")
print("="*115)
for name, d in results.items():
    print(f"{name:<65} | {d['trad_4+']:<10} | {d['seg_4+']:<12} | {d['ss_4+']:<15} | {d['any_4+']} (PE 5+: {d['pe_5+']})")
print("="*115)

