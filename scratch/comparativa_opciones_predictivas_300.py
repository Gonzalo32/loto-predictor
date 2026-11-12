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
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores
)
from cobertura_combinatoria import select_optimal_wheel_triplet
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores

def select_pool_custom(borda_scores, saturated_decades=None, max_per_decade=3, pool_size=9, strict_saturation=False):
    ranked = sorted(range(46), key=lambda x: borda_scores[x], reverse=True)
    elite_numbers = set(ranked[:2])
    pool = []
    dec_counts = Counter()
    for n in ranked:
        dec = n // 10
        limit = 2 if dec == 4 else max_per_decade
        if saturated_decades and dec in saturated_decades:
            if strict_saturation:
                limit = 1 if dec == 4 else 2
            else:
                is_elite_dec = any((e // 10 == dec) for e in elite_numbers)
                limit = 3 if is_elite_dec else min(limit, 2)
        if dec_counts[dec] < limit:
            pool.append(n)
            dec_counts[dec] += 1
            if len(pool) == pool_size: break
    if len(pool) < pool_size:
        for n in ranked:
            if n not in pool:
                pool.append(n)
                if len(pool) == pool_size: break
    return sorted(pool)

def run_experiment_300():
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    n_eval = 300
    trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
    eval_draws = trad_draws[-n_eval:]
    
    configs = {
        '1. Campeón Actual (Baseline)': {
            'use_day_seg': False, 'strict_sat': False, 'antag_threshold': 0.40, 'w_cooccur': 350.0
        },
        '2. Opción 1 (Desacoplamiento por Día)': {
            'use_day_seg': True,  'strict_sat': False, 'antag_threshold': 0.40, 'w_cooccur': 350.0
        },
        '3. Opción 2 (Anti-Saturación de Decenas Reforzada)': {
            'use_day_seg': False, 'strict_sat': True,  'antag_threshold': 0.40, 'w_cooccur': 350.0
        },
        '4. Opción 3 (Afinidad Cuántica y Antagonismo Estricto)': {
            'use_day_seg': False, 'strict_sat': False, 'antag_threshold': 0.50, 'w_cooccur': 550.0
        },
        '5. Opción 4 (Ensamble Integral: Día + Anti-Sat + Co-Ocurr)': {
            'use_day_seg': True,  'strict_sat': True,  'antag_threshold': 0.45, 'w_cooccur': 450.0
        }
    }
    
    results = {cfg: {
        'trad_4plus': 0, 'trad_3plus': 0, 'global_4plus': 0, 'global_3plus': 0,
        'pe_5plus': 0, 'dry_draws': 0, 'total_hits': 0
    } for cfg in configs}
    
    t0 = time.time()
    print("="*110)
    print(f"BENCHMARK COMPARATIVO DE OPCIONES PREDICTIVAS SOBRE {n_eval} SORTEOS CONSECUTIVOS")
    print(f"Periodo evaluado: Sorteo #{eval_draws[0]['sorteo']} ({eval_draws[0]['fecha']}) al #{eval_draws[-1]['sorteo']} ({eval_draws[-1]['fecha']})")
    print(f"Evaluando 5 configuraciones x 3 boletos = {n_eval * 3 * 5} boletos auditados...")
    print("="*110)
    
    for i, target_draw in enumerate(eval_draws):
        target_s = target_draw['sorteo']
        target_date_str = target_draw['fecha']
        target_weekday = datetime.strptime(target_date_str, '%Y-%m-%d').weekday()
        
        train_draws = [d for d in draws if d['sorteo'] < target_s]
        train_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional']
        
        # Ground truth
        all_target_draws = [d for d in draws if d['sorteo'] == target_s]
        real_by_mod = {d['modalidad']: set(d['numbers']) for d in all_target_draws}
        real_trad = real_by_mod.get('Tradicional', set())
        pozo_extra = set()
        for mod in ['Tradicional', 'Segunda', 'Revancha']:
            pozo_extra.update(real_by_mod.get(mod, set()))
            
        # Common precomputations
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
        
        # Regime
        recent_trad = train_trad[-10:]
        reps = [len(set(recent_trad[k]['numbers']) & set(recent_trad[k-1]['numbers'])) for k in range(1, len(recent_trad))]
        avg_rep = np.mean(reps) if reps else 0.8
        w_rec = 2.8 if avg_rep >= 1.0 else (1.2 if avg_rep <= 0.5 else 2.0)
        w_ddm = 1.2 if avg_rep >= 1.0 else (2.8 if avg_rep <= 0.5 else 1.8)
        
        # Base Borda
        borda_base = defaultdict(float)
        models = [(p_attn, 3.0), (p_t, w_rec), (p_cross, 1.5), (p_pcr, 1.4), (p_ddm, w_ddm)]
        for p_mod, w in models:
            rk = sorted(range(46), key=lambda x: p_mod[x], reverse=True)
            for rank, n in enumerate(rk):
                borda_base[n] += w * (45 - rank)
                
        # Saturated decades from previous date
        last_s = max(d['sorteo'] for d in train_draws)
        last_date_draws = [d for d in train_draws if d['sorteo'] == last_s]
        sat_decades = set()
        for d in last_date_draws:
            if d['modalidad'] == 'Tradicional':
                d_counts = Counter(num // 10 for num in d['numbers'])
                for dec, cnt in d_counts.items():
                    if cnt >= 3: sat_decades.add(dec)
                    
        # Day counts
        same_day_draws = [d for d in train_trad if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-35:] for n in d['numbers'])
        
        # Co-occurrence exponential matrix
        pcr_scores_exp, cooccur_mat = compute_exponential_cooccurrence_scores(train_trad, window=40, half_life=15.0)
        
        for cfg_name, cfg in configs.items():
            borda = borda_base.copy()
            
            # 1. Day of week segregation
            if cfg['use_day_seg']:
                for n in range(46):
                    borda[n] += 12.0 * counts_day[n]
                    
            # 2. Antagonist pairs
            antag = compute_antagonist_pairs(train_trad, min_expected=8.0, max_lift=cfg['antag_threshold'])
            
            # 3. Select Pools
            p9 = select_pool_custom(borda, saturated_decades=sat_decades, pool_size=9, strict_saturation=cfg['strict_sat'])
            p10 = select_pool_custom(borda, saturated_decades=sat_decades, pool_size=10, strict_saturation=cfg['strict_sat'])
            
            # Form viable combos with enhanced macro-filters
            def get_viables(pool_nums):
                v = []
                for c in itertools.combinations(pool_nums, 6):
                    s = sum(c)
                    if s < 68 or s > 202: continue
                    lows = sum(1 for n in c if n <= 22)
                    if lows < 1 or lows > 5: continue
                    dec_c = Counter(n // 10 for n in c)
                    if len(dec_c) < 3: continue
                    if dec_c[4] > 2: continue
                    if any(cnt > 3 for cnt in dec_c.values()): continue
                    if antag and any(tuple(sorted([u, v])) in antag for u, v in itertools.combinations(c, 2)): continue
                    v.append(c)
                return v if v else list(itertools.combinations(pool_nums, 6))[:10]
                
            v9 = get_viables(p9)
            v10 = get_viables(p10)
            
            w_co = cfg['w_cooccur']
            # Boleto 1: Best combo from v9
            b1 = max(v9, key=lambda c: sum(borda[n] for n in c) + w_co * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2)))
            
            # Boleto 2: Combo from v9 with overlap [2, 4] with b1
            cands_b2 = [c for c in v9 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            if not cands_b2: cands_b2 = [c for c in v9 if c != b1]
            b2 = max(cands_b2, key=lambda c: sum(borda[n] for n in c) + w_co * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2)))
            
            # Boleto 3: Combo from v10 with overlap [2, 4] with b1 and b2, triple overlap <= 2
            q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
            cands_b3 = [
                c for c in v10
                if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2
            ]
            if not cands_b3:
                cands_b3 = [c for c in v10 if c != b1 and c != b2 and len(set(c) & set(b1)) <= 4 and len(set(c) & set(b2)) <= 4]
            if not cands_b3:
                cands_b3 = [c for c in v10 if c != b1 and c != b2]
                
            b3 = max(
                cands_b3,
                key=lambda c: sum(borda[n] for n in c) + w_co * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12)
            )
            
            tickets = [b1, b2, b3]
            
            # Metrics computation
            hits_trad = [len(set(b) & real_trad) for b in tickets]
            max_t = max(hits_trad)
            if max_t >= 4: results[cfg_name]['trad_4plus'] += 1
            if max_t >= 3: results[cfg_name]['trad_3plus'] += 1
            if max_t <= 1: results[cfg_name]['dry_draws'] += 1
            
            # Global 4+ and 3+
            g4 = False
            g3 = False
            for mod, m_nums in real_by_mod.items():
                m_hits = [len(set(b) & m_nums) for b in tickets]
                if max(m_hits) >= 4: g4 = True
                if max(m_hits) >= 3: g3 = True
            if g4: results[cfg_name]['global_4plus'] += 1
            if g3: results[cfg_name]['global_3plus'] += 1
            
            # Pozo Extra
            pe_hits = [len(set(b) & pozo_extra) for b in tickets]
            if max(pe_hits) >= 5: results[cfg_name]['pe_5plus'] += 1
            results[cfg_name]['total_hits'] += sum(hits_trad)
            
        if (i + 1) % 50 == 0:
            print(f" -> Progreso: {i+1}/{n_eval} sorteos procesados ({time.time() - t0:.1f}s)")
            
    print("\n" + "="*110)
    print("RESULTADOS FINALES DEL BENCHMARK COMPARATIVO (300 SORTEOS CONSECUTIVOS)")
    print("="*110)
    print(f"{'Configuración':<48} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos(<=1)':<10}")
    print("-"*110)
    for cfg_name, r in results.items():
        print(f"{cfg_name:<48} | {r['trad_4plus']:<8} | {r['global_4plus']:<8} | {r['trad_3plus']:<8} | {r['global_3plus']:<8} | {r['pe_5plus']:<7} | {r['dry_draws']:<10}")
    print("="*110)

if __name__ == '__main__':
    run_experiment_300()
