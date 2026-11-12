import os, sys, itertools, time
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, compute_antagonist_pairs,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    es_ticket_valido
)
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores
from scratch.comparativa_opciones_predictivas_300 import select_pool_custom

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
n_eval = 300
eval_draws = trad_draws[-n_eval:]

variants = {
    '1. 5-Tiers Base (3-1-1-1)':                {'w_cooc': 0.0,  'w_day': 0.0, 'w_pairs': 0.0, 'sat_filter': False},
    '2. 5-Tiers + Saturation Filter':          {'w_cooc': 0.0,  'w_day': 0.0, 'w_pairs': 0.0, 'sat_filter': True},
    '3. 5-Tiers + Moderate Cooc (w=40)':       {'w_cooc': 40.0, 'w_day': 0.0, 'w_pairs': 0.0, 'sat_filter': True},
    '4. 5-Tiers + Moderate Cooc (w=75)':       {'w_cooc': 75.0, 'w_day': 0.0, 'w_pairs': 0.0, 'sat_filter': True},
    '5. 5-Tiers + Day-of-Week Tuning':         {'w_cooc': 0.0,  'w_day': 6.0, 'w_pairs': 0.0, 'sat_filter': True},
    '6. 5-Tiers + Pair-Coverage Maximizer':    {'w_cooc': 0.0,  'w_day': 0.0, 'w_pairs': 25.0,'sat_filter': True},
    '7. Super-Tiers Asimétrico (Day+Pairs+Sat)':{'w_cooc': 35.0, 'w_day': 6.0, 'w_pairs': 25.0,'sat_filter': True},
}

results = {v: {
    'trad_4plus': 0, 'global_4plus': 0, 'trad_3plus': 0, 'global_3plus': 0,
    'pe_5plus': 0, 'dry_draws': 0
} for v in variants}

for target_draw in eval_draws:
    target_s = target_draw['sorteo']
    target_date_str = target_draw['fecha']
    target_weekday = datetime.strptime(target_date_str, '%Y-%m-%d').weekday()
    
    train_draws = [d for d in draws if d['sorteo'] < target_s]
    train_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    
    all_target_draws = [d for d in draws if d['sorteo'] == target_s]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in all_target_draws}
    real_trad = real_by_mod.get('Tradicional', set())
    pozo_extra = set()
    for mod in ['Tradicional', 'Segunda', 'Revancha']:
        pozo_extra.update(real_by_mod.get(mod, set()))
        
    p_attn = compute_fast_attention_mlp_scores(train_trad)
    counts_t = Counter(n for d in train_trad[-60:] for n in d['numbers'])
    p_t = np.array([counts_t[n] for n in range(46)], dtype=float)
    p_t /= (p_t.sum() + 1e-6)
    
    c_cross = Counter(n for d in train_draws[-60:] for n in d['numbers'])
    p_cross = np.array([c_cross[n] for n in range(46)], dtype=float)
    p_cross /= (p_cross.sum() + 1e-6)
    
    counts_pcr = Counter(n for d in train_trad[-20:] for n in d['numbers'])
    p_pcr = np.array([counts_pcr[n] for n in range(46)], dtype=float)
    p_pcr /= (p_pcr.sum() + 1e-6)
    
    ddm = compute_delta_delay_momentum_scores(train_trad)
    p_ddm = np.array([ddm[n] for n in range(46)], dtype=float)
    p_ddm /= (p_ddm.sum() + 1e-6)
    
    recent_trad = train_trad[-10:]
    reps = [len(set(recent_trad[k]['numbers']) & set(recent_trad[k-1]['numbers'])) for k in range(1, len(recent_trad))]
    avg_rep = np.mean(reps) if reps else 0.8
    w_rec = 2.8 if avg_rep >= 1.0 else (1.2 if avg_rep <= 0.5 else 2.0)
    w_ddm = 1.2 if avg_rep >= 1.0 else (2.8 if avg_rep <= 0.5 else 1.8)
    
    borda_pure = defaultdict(float)
    models = [(p_attn, 3.0), (p_t, w_rec), (p_cross, 1.5), (p_pcr, 1.4), (p_ddm, w_ddm)]
    for p_mod, w in models:
        rk = sorted(range(46), key=lambda x: p_mod[x], reverse=True)
        for rank, n in enumerate(rk):
            borda_pure[n] += w * (45 - rank)
            
    last_s = max(d['sorteo'] for d in train_draws)
    last_date_draws = [d for d in train_draws if d['sorteo'] == last_s]
    sat_decades = set()
    for d in last_date_draws:
        if d['modalidad'] == 'Tradicional':
            d_counts = Counter(num // 10 for num in d['numbers'])
            for dec, cnt in d_counts.items():
                if cnt >= 3: sat_decades.add(dec)
                
    same_day_draws = [d for d in train_trad if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day_draws[-35:] for n in d['numbers'])
    
    _, cooccur_mat = compute_exponential_cooccurrence_scores(train_trad, window=40, half_life=15.0)
    antag = compute_antagonist_pairs(train_trad, min_expected=8.0, max_lift=0.40)
    
    for v_name, cfg in variants.items():
        w_cooc = cfg['w_cooc']
        w_day = cfg['w_day']
        w_pairs = cfg['w_pairs']
        sat_filter = cfg['sat_filter']
        
        # Build base ranking
        borda_v = borda_pure.copy()
        if sat_filter and sat_decades:
            # apply penalty to saturated decades so they don't dominate top tiers
            for n in range(46):
                if n // 10 in sat_decades:
                    borda_v[n] -= 50.0
                    
        ranked = sorted(range(46), key=lambda x: borda_v[x], reverse=True)
        t1 = ranked[0:9]
        t2 = ranked[9:18]
        t3 = ranked[18:27]
        t4 = ranked[27:36]
        
        combos = []
        for c_t1 in itertools.combinations(t1[:5], 3):
            for n2 in t2[:3]:
                for n3 in t3[:2]:
                    for n4 in t4[:2]:
                        cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                        if es_ticket_valido(cand, min_distinct_decades=3):
                            combos.append(cand)
        if len(combos) < 3:
            combos = list(itertools.combinations(t1, 6))[:10]
            
        def score_combo(c, b_dict, cooc_w):
            s = sum(b_dict[n] for n in c)
            if cooc_w > 0:
                s += cooc_w * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            return s

        combos_sorted = sorted(combos, key=lambda c: score_combo(c, borda_v, w_cooc), reverse=True)
        b1 = combos_sorted[0]
        
        # B2: optionally tuned by Day-of-Week
        if w_day > 0:
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += w_day * counts_day[n]
            combos_b2 = sorted(combos, key=lambda c: score_combo(c, borda_d, w_cooc), reverse=True)
        else:
            combos_b2 = combos_sorted
            
        b2_cands = [c for c in combos_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
        b2 = b2_cands[0] if b2_cands else combos_b2[1]
        
        # B3: optionally maximize new pairs
        p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
        b3_cands = [
            c for c in combos_sorted
            if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4
        ]
        if not b3_cands:
            b3_cands = [c for c in combos_sorted if c != b1 and c != b2]
        if not b3_cands:
            b3_cands = combos_sorted
            
        if w_pairs > 0:
            b3 = max(b3_cands, key=lambda c: score_combo(c, borda_v, w_cooc) + w_pairs * len(set(itertools.combinations(c, 2)) - p12))
        else:
            b3 = b3_cands[0]
            
        tickets = [b1, b2, b3]
        
        hits_trad = [len(set(b) & real_trad) for b in tickets]
        max_t = max(hits_trad)
        if max_t >= 4: results[v_name]['trad_4plus'] += 1
        if max_t >= 3: results[v_name]['trad_3plus'] += 1
        if max_t <= 1: results[v_name]['dry_draws'] += 1
        
        g4, g3 = False, False
        for mod, m_nums in real_by_mod.items():
            m_hits = [len(set(b) & m_nums) for b in tickets]
            if max(m_hits) >= 4: g4 = True
            if max(m_hits) >= 3: g3 = True
        if g4: results[v_name]['global_4plus'] += 1
        if g3: results[v_name]['global_3plus'] += 1
        
        pe_hits = [len(set(b) & pozo_extra) for b in tickets]
        if max(pe_hits) >= 5: results[v_name]['pe_5plus'] += 1

print("\n" + "="*115)
print(f"BENCHMARK DE OPTIMIZACIÓN 5 TIERS ({n_eval} SORTEOS CONSECUTIVOS)")
print("="*115)
print(f"{'Variante':<44} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print("-"*115)
for v_name, r in results.items():
    print(f"{v_name:<44} | {r['trad_4plus']:^8} | {r['global_4plus']:^8} | {r['trad_3plus']:^8} | {r['global_3plus']:^8} | {r['pe_5plus']:^7} | {r['dry_draws']:^8}")
print("="*115)

