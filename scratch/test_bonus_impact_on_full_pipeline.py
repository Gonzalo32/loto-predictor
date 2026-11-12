import os, sys, itertools, time
from collections import defaultdict, Counter
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
warmup = len(trad_draws) - 300
eval_draws = trad_draws[warmup:]
n_eval = len(eval_draws)

print(f"Comparando impacto de Bonus de Resonancia en {n_eval} sorteos...")

def run_test(bon_mult=1.0):
    t4_count = 0
    g4_count = 0
    t3_count = 0
    pe5_count = 0
    dry_count = 0
    
    for i, target_draw in enumerate(eval_draws):
        target_s = target_draw['sorteo']
        train_draws = [d for d in draws if d['sorteo'] < target_s]
        train_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional']
        
        last_s = max(d['sorteo'] for d in train_draws)
        last_date_draws = [d for d in train_draws if d['sorteo'] == last_s]
        res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
        
        # Fast attention
        p_attn = compute_fast_attention_mlp_scores(train_trad)
        
        # Transition probs
        counts_t = Counter(n for d in train_trad[-60:] for n in d['numbers'])
        p_t = np.array([counts_t[n] for n in range(46)], dtype=float)
        p_t /= (p_t.sum() + 1e-6)
        
        # Sub-draw cross counts
        c_cross = Counter(n for d in train_draws[-60:] for n in d['numbers'])
        p_cross = np.array([c_cross[n] for n in range(46)], dtype=float)
        p_cross /= (p_cross.sum() + 1e-6)
        
        # PCR
        counts_pcr = Counter(n for d in train_trad[-20:] for n in d['numbers'])
        p_pcr = np.array([counts_pcr[n] for n in range(46)], dtype=float)
        p_pcr /= (p_pcr.sum() + 1e-6)
        
        # DDM
        ddm = compute_delta_delay_momentum_scores(train_trad)
        p_ddm = np.array([ddm[n] for n in range(46)], dtype=float)
        p_ddm /= (p_ddm.sum() + 1e-6)
        
        # Regime
        recent_trad = train_trad[-10:]
        reps = [len(set(recent_trad[k]['numbers']) & set(recent_trad[k-1]['numbers'])) for k in range(1, len(recent_trad))]
        avg_rep = np.mean(reps) if reps else 0.8
        
        if avg_rep >= 1.0:
            w_rec, w_ddm = 2.8, 1.2
        elif avg_rep <= 0.5:
            w_rec, w_ddm = 1.2, 2.8
        else:
            w_rec, w_ddm = 2.0, 1.8
            
        borda = defaultdict(float)
        models = [(p_attn, 3.0), (p_t, w_rec), (p_cross, 1.5), (p_pcr, 1.4), (p_ddm, w_ddm)]
        for p_mod, w in models:
            rk = sorted(range(46), key=lambda x: p_mod[x], reverse=True)
            for rank, n in enumerate(rk):
                borda[n] += w * (45 - rank)
                
        for n in range(46):
            if res_counts[n] >= 2:
                borda[n] += (18.0 * bon_mult) * (res_counts[n] - 1)
            elif res_counts[n] == 1:
                borda[n] += (4.0 * bon_mult)
                
        p9 = sorted(borda.keys(), key=lambda x: borda[x], reverse=True)[:9]
        p10 = sorted(borda.keys(), key=lambda x: borda[x], reverse=True)[:10]
        
        # Form B1, B2, B3
        b1 = p9[:6]
        cands_b2 = [c for c in itertools.combinations(p9, 6) if 2 <= len(set(c) & set(b1)) <= 4 and c != tuple(b1)]
        b2 = cands_b2[0] if cands_b2 else p9[1:7]
        
        cands_b3 = [
            c for c in itertools.combinations(p10, 6)
            if 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4 and len(set(b1) & set(b2) & set(c)) <= 2
        ]
        b3 = cands_b3[0] if cands_b3 else p10[:6]
        
        real_trad = set(target_draw['numbers'])
        all_draws_target = [d for d in draws if d['sorteo'] == target_s]
        pozo_extra = set()
        for d in all_draws_target:
            if d['modalidad'] in ['Tradicional', 'Segunda', 'Revancha']:
                pozo_extra.update(d['numbers'])
                
        hits_b1 = len(set(b1) & real_trad)
        hits_b2 = len(set(b2) & real_trad)
        hits_b3 = len(set(b3) & real_trad)
        
        max_t = max(hits_b1, hits_b2, hits_b3)
        if max_t >= 4: t4_count += 1
        if max_t >= 3: t3_count += 1
        if max_t <= 1: dry_count += 1
        
        pe_hits = max(len(set(b) & pozo_extra) for b in [b1, b2, b3])
        if pe_hits >= 5: pe5_count += 1
        
        # Global 4+ in any modality
        g4 = False
        for d in all_draws_target:
            m_set = set(d['numbers'])
            if any(len(set(b) & m_set) >= 4 for b in [b1, b2, b3]):
                g4 = True
                break
        if g4: g4_count += 1
        
    return {
        'Trad 4+': t4_count,
        'Global 4+': g4_count,
        'Trad 3+': t3_count,
        'Pozo Extra 5+': pe5_count,
        'Sorteos Secos (<=1)': dry_count
    }

res_18 = run_test(bon_mult=1.0)
print(f"Bonus Actual (18 pts) : {res_18}")

res_0 = run_test(bon_mult=0.0)
print(f"Bonus Cero (0 pts)    : {res_0}")

res_half = run_test(bon_mult=0.33)
print(f"Bonus Suave (6 pts)   : {res_half}")
