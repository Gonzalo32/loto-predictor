import os, sys, itertools
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

WORKSPACE_DIR = r'c:\Users\Administrador\Desktop\lot'
sys.path.insert(0, WORKSPACE_DIR)
from generador_cobertura_ml import load_data, es_ticket_valido, compute_antagonist_pairs, compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores, select_balanced_pool_9
from cobertura_combinatoria import select_optimal_wheel_triplet
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores

draws = load_data()
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-300:]

hits_by_mod = {'Tradicional': [], 'Segunda': [], 'Revancha': [], 'SiempreSale': [], 'PozoExtra': []}
hits_4plus_by_mod = {'Tradicional': 0, 'Segunda': 0, 'Revancha': 0, 'SiempreSale': 0, 'PozoExtra': 0}
hits_3plus_by_mod = {'Tradicional': 0, 'Segunda': 0, 'Revancha': 0, 'SiempreSale': 0}

for step_i, target_draw in enumerate(eval_draws):
    target_sorteo = target_draw['sorteo']
    target_weekday = datetime.strptime(target_draw['fecha'], '%Y-%m-%d').weekday()
    
    group = [d for d in draws if d['sorteo'] == target_sorteo]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in group}
    t_trad = real_by_mod.get('Tradicional', set())
    t_seg = real_by_mod.get('Segunda', set())
    t_rev = real_by_mod.get('Revancha', set())
    t_ss = real_by_mod.get('SiempreSale', set())
    pe = t_trad | t_seg | t_rev
    
    train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
    
    counts_60 = Counter(n for d in train_draws[-60:] for n in d['numbers'])
    recency = np.zeros(46)
    for t_idx, d in enumerate(train_draws[-30:]):
        w = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency[n] += w
    pcr, cooccur = compute_exponential_cooccurrence_scores(train_draws)
    ddm = compute_delta_delay_momentum_scores(train_draws)
    attn = compute_fast_attention_mlp_scores(train_draws)
    
    trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    last_5_trad = trad_series[-5:]
    n_rep = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
    avg_rep = n_rep / 4.0
    w_rec, w_ddm = (2.8, 1.2) if avg_rep >= 1.0 else ((1.2, 2.8) if avg_rep <= 0.5 else (2.0, 1.8))
    
    last_s = max(d['sorteo'] for d in train_draws)
    last_d = [d for d in train_draws if d['sorteo'] == last_s]
    res_c = Counter(n for d in last_d for n in d['numbers'])
    prev_ss = set(n for d in last_d if d['modalidad'] == 'SiempreSale' for n in d['numbers'])
    
    borda = defaultdict(float)
    for m, w in [(counts_60, 1.5), (recency, w_rec), (pcr, 1.4), (ddm, w_ddm), ({n: attn[n] for n in range(46)}, 1.5)]:
        rk = sorted(range(46), key=lambda x: m[x], reverse=True)
        for rank, n in enumerate(rk): borda[n] += w * (45 - rank)
    for n in range(46):
        if res_c[n] >= 2: borda[n] += 18.0 * (res_c[n] - 1)
        elif res_c[n] == 1: borda[n] += 4.0
        if n in prev_ss: borda[n] += 4.0
        
    same_day = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    c_day = Counter(n for d in same_day[-40:] for n in d['numbers'])
    b_day = {n: borda[n] + 16.0 * c_day[n] for n in range(46)}
    b_hib = {n: borda[n] + 8.0 * c_day[n] for n in range(46)}
    
    p_base = select_balanced_pool_9(borda)
    p_day = select_balanced_pool_9(b_day)
    p_hib = select_balanced_pool_9(b_hib)
    
    antag = compute_antagonist_pairs(train_draws)
    recent = set(n for d in train_draws[-5:] for n in d['numbers'])
    
    def get_v(p):
        combos = list(itertools.combinations(p, 6))
        v = [c for c in combos if es_ticket_valido(c, recent, antagonist_pairs=antag, min_distinct_decades=3)]
        return v or combos[:10]
        
    v_b, v_d, v_h = get_v(p_base), get_v(p_day), get_v(p_hib)
    t_b = select_optimal_wheel_triplet(v_b, p_base, borda, cooccur)
    b1 = t_b[0]
    
    cands_b2 = [c for c in v_d if 2 <= len(set(c) & set(b1)) <= 4]
    if not cands_b2: cands_b2 = [c for c in v_d if c != b1]
    b2 = max(cands_b2, key=lambda c: sum(b_day[n] for n in c) + 350.0 * sum(cooccur[u, v] for u, v in itertools.combinations(c, 2)))
    
    q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
    cands_b3 = [c for c in v_h if 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4 and len(set(b1) & set(b2) & set(c)) <= 2]
    if not cands_b3: cands_b3 = [c for c in v_h if c != b1 and c != b2 and len(set(c) & set(b1)) <= 4 and len(set(c) & set(b2)) <= 4]
    if not cands_b3: cands_b3 = [c for c in v_h if c != b1 and c != b2]
    b3 = max(cands_b3, key=lambda c: sum(b_hib[n] for n in c) + 350.0 * sum(cooccur[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12))
    
    tickets = [b1, b2, b3]
    
    for mod_name, mod_nums in [('Tradicional', t_trad), ('Segunda', t_seg), ('Revancha', t_rev), ('SiempreSale', t_ss)]:
        if not mod_nums: continue
        max_h = max(len(set(t) & mod_nums) for t in tickets)
        hits_by_mod[mod_name].append(max_h)
        if max_h >= 3: hits_3plus_by_mod[mod_name] += 1
        if max_h >= 4: hits_4plus_by_mod[mod_name] += 1
    max_pe = max(len(set(t) & pe) for t in tickets)
    hits_by_mod['PozoExtra'].append(max_pe)
    if max_pe >= 5: hits_4plus_by_mod['PozoExtra'] += 1

print('=== DESEMPEÑO POR MODALIDAD EN 300 SORTEOS ===')
for m in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
    avg_h = np.mean(hits_by_mod[m])
    p3 = hits_3plus_by_mod[m] / len(eval_draws) * 100
    n4 = hits_4plus_by_mod[m]
    print(f'{m:<12}: Promedio {avg_h:.2f} aciertos | 3+ aciertos: {p3:.1f}% ({hits_3plus_by_mod[m]} veces) | 4+ aciertos: {n4} premios')
print(f"Pozo Extra  : Promedio {np.mean(hits_by_mod['PozoExtra']):.2f} aciertos | 5+ aciertos: {hits_4plus_by_mod['PozoExtra']} veces")

