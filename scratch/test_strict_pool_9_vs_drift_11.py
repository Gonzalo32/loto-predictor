"""
Test Comparativo: Strict Master Pool de 9/10 números vs Multi-Pool Drift (11/12 números)
Evaluado en 500 sorteos históricos consecutivos con walk-forward estricto.
"""
import os, sys, itertools, time
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_antagonist_pairs,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    select_balanced_pool_9
)
from cobertura_combinatoria import select_optimal_wheel_triplet
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores

def select_balanced_pool_9_elite(borda_scores, saturated_decades=None, max_per_decade=3):
    ranked = sorted(range(46), key=lambda x: borda_scores[x], reverse=True)
    elite_numbers = set(ranked[:2])
    pool = []
    dec_counts = Counter()
    for n in ranked:
        dec = n // 10
        limit = 2 if dec == 4 else max_per_decade
        if saturated_decades and dec in saturated_decades:
            is_elite_dec = any((e // 10 == dec) for e in elite_numbers)
            if not is_elite_dec:
                limit = min(limit, 2)
            else:
                limit = 3
        if dec_counts[dec] < limit:
            pool.append(n)
            dec_counts[dec] += 1
            if len(pool) == 9: break
    if len(pool) < 9:
        for n in ranked:
            if n not in pool:
                pool.append(n)
                if len(pool) == 9: break
    return sorted(pool)

def select_balanced_pool_10_elite(borda_scores, saturated_decades=None, max_per_decade=3):
    ranked = sorted(range(46), key=lambda x: borda_scores[x], reverse=True)
    elite_numbers = set(ranked[:2])
    pool = []
    dec_counts = Counter()
    for n in ranked:
        dec = n // 10
        limit = 2 if dec == 4 else max_per_decade
        if saturated_decades and dec in saturated_decades:
            is_elite_dec = any((e // 10 == dec) for e in elite_numbers)
            if not is_elite_dec:
                limit = min(limit, 2)
            else:
                limit = 3
        if dec_counts[dec] < limit:
            pool.append(n)
            dec_counts[dec] += 1
            if len(pool) == 10: break
    if len(pool) < 10:
        for n in ranked:
            if n not in pool:
                pool.append(n)
                if len(pool) == 10: break
    return sorted(pool)

def run_pool_concentration_benchmark(n_eval=500):
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]
    
    print("="*115)
    print(f" BENCHMARK DE CONCENTRACIÓN DE POOL: STRICT POOL 9 vs STRICT POOL 10 vs MULTI-POOL DRIFT (11)")
    print(f" Periodo: Sorteo #{eval_draws[0]['sorteo']} al #{eval_draws[-1]['sorteo']}")
    print("="*115)
    
    metrics = {
        'A. Multi-Pool Drift Actual (Union 11 numeros)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'pool_sizes': []
        },
        'B. Strict Master Pool de 9 (BIBD 2-Fold Puro)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'pool_sizes': []
        },
        'C. Strict Master Pool de 10 (Equilibrio Cobertura/Concentracion)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'pool_sizes': []
        }
    }
    
    start_time = time.time()
    
    for step_i, target_draw in enumerate(eval_draws):
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
        
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1
            
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency_scores[n] += weight
            
        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)
        attn_scores = compute_fast_attention_mlp_scores(train_draws)
        
        # Régimen dinámico cronológico
        trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
        last_5_trad = trad_series[-5:]
        n_rep = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
        avg_rep = n_rep / 4.0
        if avg_rep >= 1.0: w_rec, w_ddm = 2.8, 1.2
        elif avg_rep <= 0.5: w_rec, w_ddm = 1.2, 2.8
        else: w_rec, w_ddm = 2.0, 1.8
        
        # Resonancia sorteo previo
        last_sorteo_num = max(d['sorteo'] for d in train_draws)
        last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
        res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
        
        saturated_decades = set()
        last_trad = trad_series[-1]
        for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
            if cnt >= 3: saturated_decades.add(dec)
            
        antag = compute_antagonist_pairs(train_draws)
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
        
        borda = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda[n] += w * (45 - rank)
            
        for n in range(46):
            if res_counts[n] >= 2:
                borda[n] += 18.0 * (res_counts[n] - 1)
                
        borda_day = defaultdict(float)
        borda_hib = defaultdict(float)
        for n in range(46):
            borda_day[n] = borda[n] + 16.0 * counts_day[n]
            borda_hib[n] = borda[n] + 8.0 * counts_day[n]
            
        def get_v(p):
            c_list = list(itertools.combinations(p, 6))
            v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)]
            if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=4)]
            if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
            return v or c_list[:10]
            
        # -------------------------------------------------------------
        # ESTRATEGIA A: MULTI-POOL DRIFT (Actual de produccion)
        # -------------------------------------------------------------
        p_base_a = select_balanced_pool_9_elite(borda, saturated_decades)
        p_day_a  = select_balanced_pool_9_elite(borda_day, saturated_decades)
        p_hib_a  = select_balanced_pool_9_elite(borda_hib, saturated_decades)
        
        t_b_a = select_optimal_wheel_triplet(get_v(p_base_a), p_base_a, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        t_d_a = select_optimal_wheel_triplet(get_v(p_day_a), p_day_a, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        t_h_a = select_optimal_wheel_triplet(get_v(p_hib_a), p_hib_a, borda_hib, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        b1_a = t_b_a[0]
        b2_a = t_d_a[0] if t_d_a[0] != b1_a else t_d_a[1]
        q12_a = set(itertools.combinations(b1_a, 4)) | set(itertools.combinations(b2_a, 4))
        best_b3_a = None
        best_score_a = -float('inf')
        for c in get_v(p_hib_a):
            if c == b1_a or c == b2_a: continue
            q_c = set(itertools.combinations(c, 4))
            score_c = sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(q_c - q12_a)
            if score_c > best_score_a:
                best_score_a = score_c
                best_b3_a = c
        b3_a = best_b3_a if best_b3_a is not None else (t_h_a[0] if (t_h_a[0] != b1_a and t_h_a[0] != b2_a) else t_b_a[1])
        tickets_a = [b1_a, b2_a, b3_a]
        
        # -------------------------------------------------------------
        # ESTRATEGIA B: STRICT MASTER POOL DE 9 (BIBD 2-Fold)
        # Todos los boletos provienen del pool maestro hibrido de 9 numeros
        # -------------------------------------------------------------
        p_master_9 = select_balanced_pool_9_elite(borda_hib, saturated_decades)
        v_m9 = get_v(p_master_9)
        # B1: Máximo Borda Base
        b1_b = max(v_m9, key=lambda c: sum(borda[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        # B2: Máximo Día
        cand_b2 = [c for c in v_m9 if c != b1_b]
        b2_b = max(cand_b2 or v_m9, key=lambda c: sum(borda_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        # B3: Complementariedad Cuádruplas en Borda Híbrido
        q12_b = set(itertools.combinations(b1_b, 4)) | set(itertools.combinations(b2_b, 4))
        cand_b3 = [c for c in v_m9 if c != b1_b and c != b2_b]
        b3_b = max(cand_b3 or cand_b2 or v_m9, key=lambda c: sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12_b))
        tickets_b = [b1_b, b2_b, b3_b]
        
        # -------------------------------------------------------------
        # ESTRATEGIA C: STRICT MASTER POOL DE 10
        # -------------------------------------------------------------
        p_master_10 = select_balanced_pool_10_elite(borda_hib, saturated_decades)
        v_m10 = get_v(p_master_10)
        b1_c = max(v_m10, key=lambda c: sum(borda[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        cand_c2 = [c for c in v_m10 if c != b1_c]
        b2_c = max(cand_c2 or v_m10, key=lambda c: sum(borda_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        q12_c = set(itertools.combinations(b1_c, 4)) | set(itertools.combinations(b2_c, 4))
        cand_c3 = [c for c in v_m10 if c != b1_c and c != b2_c]
        b3_c = max(cand_c3 or cand_c2 or v_m10, key=lambda c: sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12_c))
        tickets_c = [b1_c, b2_c, b3_c]
        
        for name, t_list in [('A. Multi-Pool Drift Actual (Union 11 numeros)', tickets_a),
                             ('B. Strict Master Pool de 9 (BIBD 2-Fold Puro)', tickets_b),
                             ('C. Strict Master Pool de 10 (Equilibrio Cobertura/Concentracion)', tickets_c)]:
            u_pool = set(t_list[0]) | set(t_list[1]) | set(t_list[2])
            metrics[name]['pool_sizes'].append(len(u_pool))
            hits_trad = [len(set(t) & target_trad) for t in t_list]
            max_h_trad = max(hits_trad)
            max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
            max_pe = max(len(set(t) & pozo_extra) for t in t_list)
            
            metrics[name]['max_trad'].append(max_h_trad)
            if max_h_trad >= 3: metrics[name]['trad_3+'] += 1
            if max_h_trad >= 4: metrics[name]['trad_4+'] += 1
            if max_any >= 4: metrics[name]['any_4+'] += 1
            if max_pe >= 5: metrics[name]['pe_5+'] += 1
            
        if (step_i + 1) % 100 == 0:
            print(f" -> Progreso: {step_i + 1} de {n_eval} sorteos completados...")
            
    elapsed = time.time() - start_time
    print(f"\n[OK] Benchmark de 500 sorteos finalizado en {elapsed:.1f} segundos.\n")
    
    print("="*125)
    print(f"{'Estrategia Evaluada (500 Sorteos)':<60} | {'Pool Prom.':<10} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*125)
    for name, s in metrics.items():
        avg_p = np.mean(s['pool_sizes'])
        p3 = (s['trad_3+'] / n_eval) * 100
        n4_t = s['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = s['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = s['pe_5+']
        print(f"{name:<60} | {avg_p:<10.1f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*125)

if __name__ == '__main__':
    run_pool_concentration_benchmark(500)

