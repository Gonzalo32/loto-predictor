import os, sys, itertools
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, compute_antagonist_pairs,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    select_optimal_wheel_triplet
)
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores
from scratch.compare_600_draws_varE import select_pool, es_ticket_valido_enhanced
from scratch.test_dual_pool_architecture import select_pool_beta, compute_cross_modal_transitions

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
n_eval = 300
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]

# Test: Boleto 1 & 2 purely concentrated. Boleto 3: test 0 vs 1 vs 2 satellites from Beta.
for n_sat in [0, 1, 2]:
    t3, t4, g4, pe4, pe5 = 0, 0, 0, 0, 0
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
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
        
        cross_mat = compute_cross_modal_transitions(train_draws[-60:])
        cross_scores = np.zeros(46)
        for d_prev in last_date_draws:
            weight_m = 1.5 if d_prev['modalidad'] == 'SiempreSale' else 1.0
            for u in d_prev['numbers']:
                cross_scores += weight_m * cross_mat[u]
                
        borda = defaultdict(float)
        models = [
            (counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4),
            (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5),
            (cross_scores, 1.2)
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
        p_beta = select_pool_beta(train_draws, p_base, borda, ddm_scores, cooccur_exp_mat, size_beta=6)
        
        def get_v(p):
            c_list = list(itertools.combinations(p, 6))
            v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, antag, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
            if len(v) < 3: v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, None, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
            return v or c_list[:10]
            
        vb, vd = get_v(p_base), get_v(p_day)
        b1 = select_optimal_wheel_triplet(vb, p_base, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)[0]
        cands_b2 = [c for c in vd if 2 <= len(set(c) & set(b1)) <= 4] or [c for c in vd if c != b1]
        b2 = max(cands_b2, key=lambda c: sum(borda_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
        
        if n_sat == 0:
            vh = get_v(p_hib)
            cands_b3 = [c for c in vh if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2] or [c for c in vh if c != b1 and c != b2]
        elif n_sat == 1:
            mixed_b3 = []
            for c_alpha in itertools.combinations(p_hib, 5):
                for beta_n in p_beta:
                    tk = tuple(sorted(list(c_alpha) + [beta_n]))
                    if es_ticket_valido_enhanced(tk, recent_numbers, saturated_decades, antag, min_distinct_decades=3, filter_sum=True, filter_hilow=True):
                        mixed_b3.append(tk)
            cands_b3 = [c for c in mixed_b3 if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2] or mixed_b3[:10]
        elif n_sat == 2:
            mixed_b3 = []
            for c_alpha in itertools.combinations(p_hib, 4):
                for c_beta in itertools.combinations(p_beta, 2):
                    tk = tuple(sorted(list(c_alpha) + list(c_beta)))
                    if es_ticket_valido_enhanced(tk, recent_numbers, saturated_decades, antag, min_distinct_decades=3, filter_sum=True, filter_hilow=True):
                        mixed_b3.append(tk)
            cands_b3 = [c for c in mixed_b3 if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2] or mixed_b3[:10]
            
        b3 = max(cands_b3, key=lambda c: sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12))
        
        tickets = [b1, b2, b3]
        h_trad = max(len(set(t) & target_trad) for t in tickets)
        h_seg  = max(len(set(t) & target_seg) for t in tickets) if target_seg else 0
        h_rev  = max(len(set(t) & target_rev) for t in tickets) if target_rev else 0
        h_ss   = max(len(set(t) & target_ss) for t in tickets) if target_ss else 0
        h_pe   = max(len(set(t) & pozo_extra) for t in tickets)
        
        if h_trad >= 3: t3 += 1
        if h_trad >= 4: t4 += 1
        if max(h_trad, h_seg, h_rev, h_ss) >= 4: g4 += 1
        if h_pe >= 4: pe4 += 1
        if h_pe >= 5: pe5 += 1
        
    print(f'n_sat={n_sat} en Boleto 3 -> Trad 3+: {t3}, Trad 4+: {t4}, Global 4+: {g4}, Pozo Extra 4+: {pe4}, Pozo Extra 5+: {pe5}')

