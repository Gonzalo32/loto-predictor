import os, sys, itertools
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_antagonist_pairs,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores
)
from cobertura_combinatoria import select_optimal_wheel_triplet
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores
from scratch.compare_600_draws_varE import select_pool, es_ticket_valido_enhanced

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
n_eval = 600
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]

total_hits_base = {'trad': 0, 'seg': 0, 'rev': 0, 'ss': 0, 'pe': 0, 'max_trad_hits': []}
total_hits_opt = {'trad': 0, 'seg': 0, 'rev': 0, 'ss': 0, 'pe': 0, 'max_trad_hits': []}

for target_draw in eval_draws:
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
    
    counts_60 = Counter(n for d in train_draws[-60:] for n in d['numbers'])
    recency_scores = np.zeros(46)
    for t_idx, d in enumerate(train_draws[-30:]):
        weight = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency_scores[n] += weight
    pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
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
    same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
    
    borda = defaultdict(float)
    for m, w in [(counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
        rk = sorted(range(46), key=lambda x: m[x], reverse=True)
        for rank, n in enumerate(rk): borda[n] += w * (45 - rank)
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
    
    # 1. Base
    p_hib_base = select_pool(borda_hib, saturated_decades, pool_size=9)
    def get_v_base(p):
        c_list = list(itertools.combinations(p, 6))
        v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=3)]
        return v or c_list[:10]
    vb, vd, vh = get_v_base(p_base), get_v_base(p_day), get_v_base(p_hib_base)
    b1_b = select_optimal_wheel_triplet(vb, p_base, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)[0]
    cands_b2 = [c for c in vd if 2 <= len(set(c) & set(b1_b)) <= 4] or [c for c in vd if c != b1_b]
    b2_b = max(cands_b2, key=lambda c: sum(borda_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
    q12_b = set(itertools.combinations(b1_b, 4)) | set(itertools.combinations(b2_b, 4))
    cands_b3 = [c for c in vh if (2 <= len(set(c) & set(b1_b)) <= 4) and (2 <= len(set(c) & set(b2_b)) <= 4) and len(set(b1_b) & set(b2_b) & set(c)) <= 2] or [c for c in vh if c != b1_b and c != b2_b]
    b3_b = max(cands_b3, key=lambda c: sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12_b))
    
    # 2. Opt
    p_hib_opt = select_pool(borda_hib, saturated_decades, pool_size=10)
    def get_v_opt(p):
        c_list = list(itertools.combinations(p, 6))
        v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, antag, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
        if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, None, min_distinct_decades=3)]
        return v or c_list[:10]
    vbo, vdo, vho = get_v_opt(p_base), get_v_opt(p_day), get_v_opt(p_hib_opt)
    b1_o = select_optimal_wheel_triplet(vbo, p_base, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)[0]
    cands_b2o = [c for c in vdo if 2 <= len(set(c) & set(b1_o)) <= 4] or [c for c in vdo if c != b1_o]
    b2_o = max(cands_b2o, key=lambda c: sum(borda_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
    q12_o = set(itertools.combinations(b1_o, 4)) | set(itertools.combinations(b2_o, 4))
    cands_b3o = [c for c in vho if (2 <= len(set(c) & set(b1_o)) <= 4) and (2 <= len(set(c) & set(b2_o)) <= 4) and len(set(b1_o) & set(b2_o) & set(c)) <= 2] or [c for c in vho if c != b1_o and c != b2_o]
    b3_o = max(cands_b3o, key=lambda c: sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12_o))
    
    t_base = [b1_b, b2_b, b3_b]
    t_opt = [b1_o, b2_o, b3_o]
    
    total_hits_base['trad'] += sum(len(set(t) & target_trad) for t in t_base)
    total_hits_base['seg']  += sum(len(set(t) & target_seg) for t in t_base)
    total_hits_base['rev']  += sum(len(set(t) & target_rev) for t in t_base)
    total_hits_base['ss']   += sum(len(set(t) & target_ss) for t in t_base)
    total_hits_base['pe']   += sum(len(set(t) & pozo_extra) for t in t_base)
    total_hits_base['max_trad_hits'].append(max(len(set(t) & target_trad) for t in t_base))
    
    total_hits_opt['trad'] += sum(len(set(t) & target_trad) for t in t_opt)
    total_hits_opt['seg']  += sum(len(set(t) & target_seg) for t in t_opt)
    total_hits_opt['rev']  += sum(len(set(t) & target_rev) for t in t_opt)
    total_hits_opt['ss']   += sum(len(set(t) & target_ss) for t in t_opt)
    total_hits_opt['pe']   += sum(len(set(t) & pozo_extra) for t in t_opt)
    total_hits_opt['max_trad_hits'].append(max(len(set(t) & target_trad) for t in t_opt))

tot_b = total_hits_base['trad'] + total_hits_base['seg'] + total_hits_base['rev'] + total_hits_base['ss']
tot_o = total_hits_opt['trad'] + total_hits_opt['seg'] + total_hits_opt['rev'] + total_hits_opt['ss']

print("="*80)
print("AUDITORIA EXACTA DE BOLILLAS ACERTADAS (600 SORTEOS / 1.800 BOLETOS)")
print("="*80)
print(f"TRADICIONAL : Base = {total_hits_base['trad']} bolillas  |  Optimizado = {total_hits_opt['trad']} bolillas (Diferencia: +{total_hits_opt['trad'] - total_hits_base['trad']})")
print(f"SEGUNDA     : Base = {total_hits_base['seg']} bolillas  |  Optimizado = {total_hits_opt['seg']} bolillas (Diferencia: +{total_hits_opt['seg'] - total_hits_base['seg']})")
print(f"REVANCHA    : Base = {total_hits_base['rev']} bolillas  |  Optimizado = {total_hits_opt['rev']} bolillas (Diferencia: +{total_hits_opt['rev'] - total_hits_base['rev']})")
print(f"SIEMPRE SALE: Base = {total_hits_base['ss']} bolillas  |  Optimizado = {total_hits_opt['ss']} bolillas (Diferencia: +{total_hits_opt['ss'] - total_hits_base['ss']})")
print(f"POZO EXTRA  : Base = {total_hits_base['pe']} bolillas  |  Optimizado = {total_hits_opt['pe']} bolillas (Diferencia: +{total_hits_opt['pe'] - total_hits_base['pe']})")
print("-"*80)
print(f"TOTAL GLOBAL BOLILLAS: Base = {tot_b}  |  Optimizado = {tot_o} (Diferencia: +{tot_o - tot_b} bolillas)")
print("="*80)

