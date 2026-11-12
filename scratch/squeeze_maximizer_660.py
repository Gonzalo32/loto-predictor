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
n_eval = 660
eval_draws = trad_draws[-n_eval:]

configs = {
    'H4: Tier 1 Expandido':          {'mode': 'h4'},
    'H7: Fusión Suprema':            {'mode': 'h7'},
    'S3: Hybrid Lock (H4+H7)':       {'mode': 's3_hybrid'},
    'S4: Hybrid Lock + Parity Filter':{'mode': 's4_parity'},
    'S5: Expanded Degrees (6-4-3-3)':{'mode': 's5_expanded'},
    'S6: Ultra-Pareto Squeeze':      {'mode': 's6_pareto'}
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
    
    borda_v = borda_pure.copy()
    if sat_decades:
        for n in range(46):
            if n // 10 in sat_decades: borda_v[n] -= 50.0
            
    ranked = sorted(range(46), key=lambda x: borda_v[x], reverse=True)
    t1 = ranked[0:9]
    t2 = ranked[9:18]
    t3 = ranked[18:27]
    t4 = ranked[27:36]

    for c_name, c_cfg in configs.items():
        mode = c_cfg['mode']
        
        if mode == 'h4':
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
            b3 = max(b3_cands, key=lambda c: sc_h4(c, borda_v) + 25.0 * len(set(itertools.combinations(c, 2)) - p12)) if b3_cands else combos_sorted[2]
            tickets = [b1, b2, b3]

        elif mode == 'h7':
            combos_b1 = []
            for c_t1 in itertools.combinations(t1[:5], 3):
                for c_t2 in itertools.combinations(t2[:4], 2):
                    for n3 in t3[:3]:
                        cand = tuple(sorted(list(c_t1) + list(c_t2) + [n3]))
                        if es_ticket_valido(cand, min_distinct_decades=3): combos_b1.append(cand)
            b1 = max(combos_b1, key=lambda c: sum(borda_v[n] for n in c))
            combos_b2 = []
            for c_t1 in itertools.combinations(t1[:5], 3):
                for n2 in t2[:3]:
                    for n3 in t3[:2]:
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos_b2.append(cand)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            b2_cands = [c for c in combos_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(b2_cands, key=lambda c: sum(borda_d[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2)))
            combos_b3 = []
            for c_t1 in itertools.combinations(t1[:5], 2):
                for n2 in t2[:3]:
                    for c_t3 in itertools.combinations(t3[:3], 2):
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2] + list(c_t3) + [n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos_b3.append(cand)
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            b3 = max(b3_cands, key=lambda c: sum(borda_v[n] for n in c) + 30.0 * len(set(itertools.combinations(c, 2)) - p12))
            tickets = [b1, b2, b3]

        elif mode == 's3_hybrid':
            # S3: B1 & B2 from H4 (locking the 16 4+ core), B3 from H7 (wide-net pair maximizer)
            combos_h4 = []
            for c_t1 in itertools.combinations(t1[:6], 3):
                for n2 in t2[:3]:
                    for n3 in t3[:2]:
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos_h4.append(cand)
            def sc_s3(c, b_dict):
                return sum(b_dict[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            combos_h4_sorted = sorted(combos_h4, key=lambda c: sc_s3(c, borda_v), reverse=True)
            b1 = combos_h4_sorted[0]
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            combos_h4_b2 = sorted(combos_h4, key=lambda c: sc_s3(c, borda_d), reverse=True)
            b2_cands = [c for c in combos_h4_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = b2_cands[0] if b2_cands else combos_h4_b2[1]
            
            # B3: from H7 asymmetric complementary generator
            combos_b3 = []
            for c_t1 in itertools.combinations(t1[:5], 2):
                for n2 in t2[:3]:
                    for c_t3 in itertools.combinations(t3[:3], 2):
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2] + list(c_t3) + [n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos_b3.append(cand)
            if not combos_b3: combos_b3 = combos_h4
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not b3_cands: b3_cands = [c for c in combos_b3 if c != b1 and c != b2]
            b3 = max(b3_cands, key=lambda c: sum(borda_v[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2)) + 30.0 * len(set(itertools.combinations(c, 2)) - p12))
            tickets = [b1, b2, b3]

        elif mode == 's4_parity':
            # S4: S3 + Parity filter (evens in [2, 4])
            def valid_parity(c):
                ev = sum(1 for n in c if n % 2 == 0)
                return 2 <= ev <= 4
            combos_h4 = []
            for c_t1 in itertools.combinations(t1[:6], 3):
                for n2 in t2[:3]:
                    for n3 in t3[:2]:
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3) and valid_parity(cand): combos_h4.append(cand)
            if not combos_h4:
                combos_h4 = [c for c in itertools.combinations(t1, 6) if es_ticket_valido(c)]
            def sc_s4(c, b_dict):
                return sum(b_dict[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            combos_h4_sorted = sorted(combos_h4, key=lambda c: sc_s4(c, borda_v), reverse=True)
            b1 = combos_h4_sorted[0]
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            combos_h4_b2 = sorted(combos_h4, key=lambda c: sc_s4(c, borda_d), reverse=True)
            b2_cands = [c for c in combos_h4_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = b2_cands[0] if b2_cands else combos_h4_b2[1]
            
            combos_b3 = []
            for c_t1 in itertools.combinations(t1[:5], 2):
                for n2 in t2[:3]:
                    for c_t3 in itertools.combinations(t3[:3], 2):
                        for n4 in t4[:2]:
                            cand = tuple(sorted(list(c_t1) + [n2] + list(c_t3) + [n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3) and valid_parity(cand): combos_b3.append(cand)
            if not combos_b3: combos_b3 = combos_h4
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not b3_cands: b3_cands = [c for c in combos_b3 if c != b1 and c != b2]
            b3 = max(b3_cands, key=lambda c: sum(borda_v[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2)) + 30.0 * len(set(itertools.combinations(c, 2)) - p12))
            tickets = [b1, b2, b3]

        elif mode == 's5_expanded':
            # Expanded degrees of freedom: t1[:6], t2[:4], t3[:3], t4[:3]
            combos_exp = []
            for c_t1 in itertools.combinations(t1[:6], 3):
                for n2 in t2[:4]:
                    for n3 in t3[:3]:
                        for n4 in t4[:3]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos_exp.append(cand)
            def sc_s5(c, b_dict):
                return sum(b_dict[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            combos_exp_sorted = sorted(combos_exp, key=lambda c: sc_s5(c, borda_v), reverse=True)
            b1 = combos_exp_sorted[0]
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            combos_b2 = sorted(combos_exp, key=lambda c: sc_s5(c, borda_d), reverse=True)
            b2_cands = [c for c in combos_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = b2_cands[0] if b2_cands else combos_b2[1]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_exp_sorted if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            b3 = max(b3_cands, key=lambda c: sc_s5(c, borda_v) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if b3_cands else combos_exp_sorted[2]
            tickets = [b1, b2, b3]

        elif mode == 's6_pareto':
            # Ultra-Pareto: B1 from S5 (6-4-3-3), B2 Day-tuned from S5, B3 Asymmetric from H7 with pairs=35
            combos_exp = []
            for c_t1 in itertools.combinations(t1[:6], 3):
                for n2 in t2[:4]:
                    for n3 in t3[:3]:
                        for n4 in t4[:3]:
                            cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos_exp.append(cand)
            def sc_s6(c, b_dict):
                return sum(b_dict[n] for n in c) + 40.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            combos_exp_sorted = sorted(combos_exp, key=lambda c: sc_s6(c, borda_v), reverse=True)
            b1 = combos_exp_sorted[0]
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 8.0 * counts_day[n]
            combos_b2 = sorted(combos_exp, key=lambda c: sc_s6(c, borda_d), reverse=True)
            b2_cands = [c for c in combos_b2 if 3 <= len(set(c) & set(b1)) <= 4 and c != b1]
            if not b2_cands: b2_cands = [c for c in combos_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = b2_cands[0] if b2_cands else combos_b2[1]
            
            combos_b3 = []
            for c_t1 in itertools.combinations(t1[:5], 2):
                for n2 in t2[:4]:
                    for c_t3 in itertools.combinations(t3[:3], 2):
                        for n4 in t4[:3]:
                            cand = tuple(sorted(list(c_t1) + [n2] + list(c_t3) + [n4]))
                            if es_ticket_valido(cand, min_distinct_decades=3): combos_b3.append(cand)
            if not combos_b3: combos_b3 = combos_exp
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            b3_cands = [c for c in combos_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not b3_cands: b3_cands = [c for c in combos_b3 if c != b1 and c != b2]
            b3 = max(b3_cands, key=lambda c: sum(borda_v[n] for n in c) + 40.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2)) + 35.0 * len(set(itertools.combinations(c, 2)) - p12))
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
print('EVALUACION HISTORICA COMPLETA: SQUEEZE MAXIMIZER (660 SORTEOS / 1.980 BOLETOS)')
print('='*105)
print(f"{'Configuración':<36} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print('-'*105)
for c, r in results.items():
    print(f"{c:<36} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print('='*105)

