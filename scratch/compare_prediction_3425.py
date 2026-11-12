import os, sys
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import itertools

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, compute_antagonist_pairs,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    select_optimal_wheel_triplet
)
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores
from scratch.compare_600_draws_varE import select_pool, es_ticket_valido_enhanced

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))

def compute_cross_modal_transitions(train_draws, window=60):
    sub = train_draws[-window:] if window else train_draws
    trans_matrix = np.zeros((46, 46))
    for i in range(1, len(sub)):
        for u in sub[i-1]['numbers']:
            for v in sub[i]['numbers']:
                trans_matrix[u][v] += 1.0
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return trans_matrix / row_sums

# Run for next draw: Sorteo 3425 (Wednesday 11/11/2026)
train_draws = draws
counts_60 = Counter(n for d in train_draws[-60:] for n in d['numbers'])
recency_scores = np.zeros(46)
for t_idx, d in enumerate(train_draws[-30:]):
    weight = np.exp((t_idx - 30) / 10.0)
    for n in d['numbers']: recency_scores[n] += weight
    
pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws, window=40)
ddm_scores = compute_delta_delay_momentum_scores(train_draws)
attn_scores = compute_fast_attention_mlp_scores(train_draws)

trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
last_5_trad = trad_series[-5:]
n_rep = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
avg_rep = n_rep / 4.0
w_rec, w_ddm = (2.8, 1.2) if avg_rep >= 1.0 else ((1.2, 2.8) if avg_rep <= 0.5 else (2.0, 1.8))

last_sorteo_num = max(d['sorteo'] for d in train_draws)
last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
prev_ss_nums = set(n for d in last_date_draws if d['modalidad'] == 'SiempreSale' for n in d['numbers'])

saturated_decades = set()
for dec, cnt in Counter(n // 10 for n in trad_series[-1]['numbers']).items():
    if cnt >= 3: saturated_decades.add(dec)
    
antag = compute_antagonist_pairs(train_draws)
recent_numbers = set(n for d in train_draws[-5:] for n in d['numbers'])
target_weekday = 2 # Wednesday
same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])

cross_mat = compute_cross_modal_transitions(train_draws, window=60)
cross_scores = np.zeros(46)
for d_prev in last_date_draws:
    weight_m = 1.5 if d_prev['modalidad'] == 'SiempreSale' else 1.0
    for u in d_prev['numbers']:
        cross_scores += weight_m * cross_mat[u]

schemes = {
    'Baseline Actual': {
        'w_attn': 1.5, 'w_cross': 1.2, 'w_pcr': 1.4, 'w_counts': 1.5, 'mult_rec': 1.0, 'mult_ddm': 1.0
    },
    'Scheme 3 (Inercia Dinamica)': {
        'w_attn': 2.0, 'w_cross': 1.5, 'w_pcr': 1.2, 'w_counts': 1.2, 'mult_rec': 1.3, 'mult_ddm': 0.8
    }
}

for name, s in schemes.items():
    borda = defaultdict(float)
    models = [
        (counts_60, s['w_counts']),
        (recency_scores, w_rec * s['mult_rec']),
        (pcr_scores, s['w_pcr']),
        (ddm_scores, w_ddm * s['mult_ddm']),
        ({n: attn_scores[n] for n in range(46)}, s['w_attn']),
        (cross_scores, s['w_cross'])
    ]
    for m, w in models:
        rk = sorted(range(46), key=lambda x: m[x], reverse=True)
        for rank, n in enumerate(rk):
            borda[n] += w * (45 - rank)
            
    for n in range(46):
        if res_counts[n] >= 2: borda[n] += 18.0 * (res_counts[n] - 1)
        elif res_counts[n] == 1: borda[n] += 4.0
        if n in prev_ss_nums: borda[n] += 4.0
        
    borda_day = defaultdict(float)
    borda_hib = defaultdict(float)
    for n in range(46):
        borda_day[n] = borda[n] + 16.0 * counts_day[n]
        borda_hib[n] = borda[n] + 8.0 * counts_day[n]
        
    p_base = select_pool(borda, saturated_decades, pool_size=9)
    p_day = select_pool(borda_day, saturated_decades, pool_size=9)
    p_hib = select_pool(borda_hib, saturated_decades, pool_size=10)
    
    def get_v(p):
        c_list = list(itertools.combinations(p, 6))
        v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, antag, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
        if len(v) < 3: v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, None, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
        return v or c_list[:10]
        
    vb, vd, vh = get_v(p_base), get_v(p_day), get_v(p_hib)
    b1 = select_optimal_wheel_triplet(vb, p_base, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)[0]
    cands_b2 = [c for c in vd if 2 <= len(set(c) & set(b1)) <= 4] or [c for c in vd if c != b1]
    b2 = max(cands_b2, key=lambda c: sum(borda_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
    q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
    cands_b3 = [c for c in vh if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2] or [c for c in vh if c != b1 and c != b2 and len(set(c) & set(b1)) <= 4 and len(set(c) & set(b2)) <= 4] or [c for c in vh if c != b1 and c != b2]
    b3 = max(cands_b3, key=lambda c: sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12))
    
    print(f"\n--- {name} (Sorteo 3425) ---")
    print(f"Pool Base (9): {p_base}")
    print(f"Pool Dia  (9): {p_day}")
    print(f"Pool Hib (10): {p_hib}")
    print(f"Boleto 1: {sorted(b1)}")
    print(f"Boleto 2: {sorted(b2)}")
    print(f"Boleto 3: {sorted(b3)}")
    all_u = sorted(list(set(b1) | set(b2) | set(b3)))
    print(f"Total unicos ({len(all_u)}): {all_u}")

