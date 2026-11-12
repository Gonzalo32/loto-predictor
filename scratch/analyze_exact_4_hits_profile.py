import os, sys, itertools
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)
from generador_cobertura_ml import load_data, compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores, select_optimal_wheel_triplet
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores
from scratch.compare_600_draws_varE import select_pool, es_ticket_valido_enhanced

draws = load_data()
trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
n_eval = 600
eval_draws = trad_draws[-n_eval:]

hit_4_draws = []
hit_3_draws = []

for step_i, target_draw in enumerate(eval_draws):
    target_sorteo = target_draw['sorteo']
    target_trad = set(target_draw['numbers'])
    train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
    
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
        
    antag = set()
    recent_numbers = set(n for d in train_draws[-5:] for n in d['numbers'])
    target_weekday = datetime.strptime(target_draw['fecha'], '%Y-%m-%d').weekday()
    same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
    
    borda = defaultdict(float)
    models = [
        (counts_60, 1.2),
        (recency_scores, w_rec * 1.3),
        (pcr_scores, 1.4),
        (ddm_scores, w_ddm * 0.8),
        ({n: attn_scores[n] for n in range(46)}, 2.0)
    ]
    for m, w in models:
        rk = sorted(range(46), key=lambda x: m[x], reverse=True)
        for rank, n in enumerate(rk):
            borda[n] += w * (45 - rank)
            
    for n in range(46):
        if res_counts[n] >= 2: borda[n] += 18.0 * (res_counts[n] - 1)
        elif res_counts[n] == 1: borda[n] += 4.0
        if n in prev_ss_nums: borda[n] += 4.0
        
    p_base = select_pool(borda, saturated_decades, pool_size=9)
    in_pool = len(set(p_base) & target_trad)
    
    if in_pool >= 4:
        hit_4_draws.append((target_sorteo, in_pool, set(p_base) & target_trad))
    elif in_pool == 3:
        hit_3_draws.append(target_sorteo)

print(f"Total sorteos donde el Pool Base de 9 metio 4 o mas numeros ganadores: {len(hit_4_draws)} ({len(hit_4_draws)/n_eval*100:.1f}%)")
print(f"Total sorteos donde el Pool Base de 9 metio exactamente 3 numeros ganadores: {len(hit_3_draws)} ({len(hit_3_draws)/n_eval*100:.1f}%)")
print(f"Ejemplos de sorteos con 4+ en el Pool Base:")
for s, cnt, nums in hit_4_draws[:10]:
    print(f"  Sorteo #{s}: {cnt} aciertos en pool -> {sorted(list(nums))}")

