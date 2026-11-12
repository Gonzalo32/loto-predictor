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

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
n_eval = 300
eval_draws = trad_draws[-n_eval:]

# Test new optimization hypotheses:
# H1: V7 (Current best)
# H2: Explicit DDM for Tier 4 (pure delay momentum instead of rank 28-36)
# H3: Asymmetric Tier Allocation (B1: 3 T1 + 2 T2 + 1 T3; B2: 3 T1 + 1 T2 + 1 T3 + 1 T4; B3: 2 T1 + 1 T2 + 2 T3 + 1 T4)
# H4: Expanded Tier 1 (t1[:6] instead of t1[:5])
# H5: Optimal Co-occurrence & Pair Weight Blend (w_cooc=20.0, w_day=5.0, w_pairs=30.0)
# H6: Unified Champion-Tiers Fusion (B1 from V7, B2 from V7 day-tuned, B3 with Pair-Coverage & DDM)

configs = {
    'V7 (Actual)':                {'mode': 'v7'},
    'H2: Tier 4 DDM Explicito':   {'mode': 'h2_ddm'},
    'H3: Asignacion Asimetrica':  {'mode': 'h3_asym'},
    'H4: Tier 1 Expandido (T1=6)':{'mode': 'h4_exp_t1'},
    'H5: Blend Cooc=20 Pairs=30': {'mode': 'h5_blend'},
    'H6: Fusion V6+V7 Sinergica': {'mode': 'h6_fusion'}
}

results = {c: {'trad_4': 0, 'glob_4': 0, 'trad_3': 0, 'glob_3': 0, 'pe_5': 0, 'secos': 0} for c in configs}

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
    
    borda_v = borda_pure.copy()
    if sat_decades:
        for n in range(46):
            if n // 10 in sat_decades: borda_v[n] -= 50.0
            
    ranked = sorted(range(46), key=lambda x: borda_v[x], reverse=True)
    t1 = ranked[0:9]
    t2 = ranked[9:18]
    t3 = ranked[18:27]
    t4 = ranked[27:36]
    
    # Ranked by DDM specifically for delay tier
    ranked_ddm = sorted(range(46), key=lambda x: p_ddm[x], reverse=True)
    t4_ddm = [n for n in ranked_ddm if n not in t1[:6] and n not in t2[:4]][:6]

    for c_name, c_cfg in configs.items():
        mode = c_cfg['mode']
        
        if mode == 'v7':
            combos = []
            for c_t1 in itertools.combinations(t1[:5], 3):
                for n2 in t2[:3]:
                    for n3 in t3[:2]:
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos.append(cand)
            if len(combos) < 3: combos = list(itertools.combinations(t1, 6))[:10]
            def sc_v7(c, b_dict):
                return sum(b_dict[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            combos_sorted = sorted(combos, key=lambda c: sc_v7(c, borda_v), reverse=True)
            b1 = combos_sorted[0]
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            combos_b2 = sorted(combos, key=lambda c: sc_v7(c, borda_d), reverse=True)
            b2_cands = [c for c in combos_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = b2_cands[0] if b2_cands else combos_b2[1]
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_sorted if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not b3_cands: b3_cands = [c for c in combos_sorted if c != b1 and c != b2]
            if not b3_cands: b3_cands = combos_sorted
            b3 = max(b3_cands, key=lambda c: sc_v7(c, borda_v) + 25.0 * len(set(itertools.combinations(c, 2)) - p12))
            tickets = [b1, b2, b3]

        elif mode == 'h2_ddm':
            # Use t4_ddm instead of standard t4
            combos = []
            for c_t1 in itertools.combinations(t1[:5], 3):
                for n2 in t2[:3]:
                    for n3 in t3[:2]:
                        for n4 in t4_ddm[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos.append(cand)
            if len(combos) < 3: combos = list(itertools.combinations(t1, 6))[:10]
            def sc_h2(c, b_dict):
                return sum(b_dict[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            combos_sorted = sorted(combos, key=lambda c: sc_h2(c, borda_v), reverse=True)
            b1 = combos_sorted[0]
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            combos_b2 = sorted(combos, key=lambda c: sc_h2(c, borda_d), reverse=True)
            b2_cands = [c for c in combos_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = b2_cands[0] if b2_cands else combos_b2[1]
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_sorted if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not b3_cands: b3_cands = [c for c in combos_sorted if c != b1 and c != b2]
            if not b3_cands: b3_cands = combos_sorted
            b3 = max(b3_cands, key=lambda c: sc_h2(c, borda_v) + 25.0 * len(set(itertools.combinations(c, 2)) - p12))
            tickets = [b1, b2, b3]

        elif mode == 'h3_asym':
            # B1: 3 de T1 + 2 de T2 + 1 de T3 (Pure Hot/Momentum)
            combos_b1 = []
            for c_t1 in itertools.combinations(t1[:5], 3):
                for c_t2 in itertools.combinations(t2[:4], 2):
                    for n3 in t3[:3]:
                        cand = tuple(sorted(list(c_t1) + list(c_t2) + [n3]))
                        if es_ticket_valido(cand, min_distinct_decades=3): combos_b1.append(cand)
            if not combos_b1: combos_b1 = list(itertools.combinations(t1, 6))[:10]
            def sc_h3(c, b_dict):
                return sum(b_dict[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            b1 = max(combos_b1, key=lambda c: sc_h3(c, borda_v))
            
            # B2: 3 de T1 + 1 de T2 + 1 de T3 + 1 de T4 (Standard Balanced + Day)
            combos_b2 = []
            for c_t1 in itertools.combinations(t1[:5], 3):
                for n2 in t2[:3]:
                    for n3 in t3[:2]:
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos_b2.append(cand)
            if not combos_b2: combos_b2 = combos_b1
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            b2_cands = [c for c in combos_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(b2_cands, key=lambda c: sc_h3(c, borda_d)) if b2_cands else combos_b2[0]
            
            # B3: 2 de T1 + 1 de T2 + 2 de T3 + 1 de T4 (Complementary + Pairs)
            combos_b3 = []
            for c_t1 in itertools.combinations(t1[:5], 2):
                for n2 in t2[:3]:
                    for c_t3 in itertools.combinations(t3[:3], 2):
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2] + list(c_t3) + [n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos_b3.append(cand)
            if not combos_b3: combos_b3 = combos_b2
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not b3_cands: b3_cands = [c for c in combos_b3 if c != b1 and c != b2]
            if not b3_cands: b3_cands = combos_b3
            b3 = max(b3_cands, key=lambda c: sc_h3(c, borda_v) + 25.0 * len(set(itertools.combinations(c, 2)) - p12))
            tickets = [b1, b2, b3]

        elif mode == 'h4_exp_t1':
            # Expanded T1: take from t1[:6]
            combos = []
            for c_t1 in itertools.combinations(t1[:6], 3):
                for n2 in t2[:3]:
                    for n3 in t3[:2]:
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos.append(cand)
            if len(combos) < 3: combos = list(itertools.combinations(t1, 6))[:10]
            def sc_h4(c, b_dict):
                return sum(b_dict[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            combos_sorted = sorted(combos, key=lambda c: sc_h4(c, borda_v), reverse=True)
            b1 = combos_sorted[0]
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            combos_b2 = sorted(combos, key=lambda c: sc_h4(c, borda_d), reverse=True)
            b2_cands = [c for c in combos_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = b2_cands[0] if b2_cands else combos_b2[1]
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_sorted if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not b3_cands: b3_cands = [c for c in combos_sorted if c != b1 and c != b2]
            if not b3_cands: b3_cands = combos_sorted
            b3 = max(b3_cands, key=lambda c: sc_h4(c, borda_v) + 25.0 * len(set(itertools.combinations(c, 2)) - p12))
            tickets = [b1, b2, b3]

        elif mode == 'h5_blend':
            # w_cooc = 20.0, w_pairs = 30.0, w_day = 5.0
            combos = []
            for c_t1 in itertools.combinations(t1[:5], 3):
                for n2 in t2[:3]:
                    for n3 in t3[:2]:
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos.append(cand)
            if len(combos) < 3: combos = list(itertools.combinations(t1, 6))[:10]
            def sc_h5(c, b_dict):
                return sum(b_dict[n] for n in c) + 20.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            combos_sorted = sorted(combos, key=lambda c: sc_h5(c, borda_v), reverse=True)
            b1 = combos_sorted[0]
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 5.0 * counts_day[n]
            combos_b2 = sorted(combos, key=lambda c: sc_h5(c, borda_d), reverse=True)
            b2_cands = [c for c in combos_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = b2_cands[0] if b2_cands else combos_b2[1]
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_sorted if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not b3_cands: b3_cands = [c for c in combos_sorted if c != b1 and c != b2]
            if not b3_cands: b3_cands = combos_sorted
            b3 = max(b3_cands, key=lambda c: sc_h5(c, borda_v) + 30.0 * len(set(itertools.combinations(c, 2)) - p12))
            tickets = [b1, b2, b3]

        elif mode == 'h6_fusion':
            # Fusion: B1 from pure V6 (cooc=0, max borda), B2 from V7 (cooc=35, day-tuned), B3 pair maximizer (pairs=30)
            combos = []
            for c_t1 in itertools.combinations(t1[:5], 3):
                for n2 in t2[:3]:
                    for n3 in t3[:2]:
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos.append(cand)
            if len(combos) < 3: combos = list(itertools.combinations(t1, 6))[:10]
            
            # B1: Pure Borda (as in V6)
            b1 = max(combos, key=lambda c: sum(borda_v[n] for n in c))
            
            # B2: Day-tuned + Cooc (as in V7)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            b2_cands = [c for c in combos if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(b2_cands, key=lambda c: sum(borda_d[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))) if b2_cands else combos[0]
            
            # B3: Maximize Uncovered Pairs (w_pairs=30.0)
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not b3_cands: b3_cands = [c for c in combos if c != b1 and c != b2]
            if not b3_cands: b3_cands = combos
            b3 = max(b3_cands, key=lambda c: sum(borda_v[n] for n in c) + 30.0 * len(set(itertools.combinations(c, 2)) - p12))
            tickets = [b1, b2, b3]

        hits_trad = [len(set(b) & real_trad) for b in tickets]
        max_t = max(hits_trad)
        if max_t >= 4: results[c_name]['trad_4'] += 1
        if max_t >= 3: results[c_name]['trad_3'] += 1
        if max_t <= 1: results[c_name]['secos'] += 1
        
        g4, g3 = False, False
        for mod, m_nums in real_by_mod.items():
            m_hits = [len(set(b) & m_nums) for b in tickets]
            if max(m_hits) >= 4: g4 = True
            if max(m_hits) >= 3: g3 = True
        if g4: results[c_name]['glob_4'] += 1
        if g3: results[c_name]['glob_3'] += 1
        
        pe_hits = [len(set(b) & pozo_extra) for b in tickets]
        if max(pe_hits) >= 5: results[c_name]['pe_5'] += 1

print('='*105)
print(f'BENCHMARK DE BUSQUEDA PROFUNDA ({n_eval} SORTEOS CONSECUTIVOS)')
print('='*105)
print(f"{'Configuración':<32} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print('-'*105)
for c, r in results.items():
    print(f"{c:<32} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print('='*105)

