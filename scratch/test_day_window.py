"""
Test: Optimización de la Ventana de Análisis por Día de la Semana
Comparativa: Ventana 40 sub-sorteos (10 fechas) vs Ventana 80 sub-sorteos (20 fechas) vs Ventana 120 (30 fechas)
Evaluado en 500 sorteos con walk-forward estricto.
"""
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
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    select_balanced_pool_9
)
from cobertura_combinatoria import select_optimal_wheel_triplet
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores

def select_pool_9_elite(borda_scores, saturated_decades=None, max_per_decade=3):
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

def run_day_window_test(n_eval=500):
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]
    
    print("="*115)
    print(f" BENCHMARK: VENTANA DE DÍA DE SEMANA (40 vs 80 vs 120 sub-sorteos) EN {n_eval} SORTEOS")
    print("="*115)
    
    metrics = {
        '1. Ventana 40 (10 semanas)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []},
        '2. Ventana 80 (20 semanas)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []},
        '3. Ventana 120 (30 semanas)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []}
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
        
        counts_60 = Counter(n for d in train_draws[-60:] for n in d['numbers'])
        recency = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency[n] += weight
            
        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)
        attn_scores = compute_fast_attention_mlp_scores(train_draws)
        
        trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
        last_5_trad = trad_series[-5:]
        n_rep = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
        avg_rep = n_rep / 4.0
        w_rec, w_ddm = (2.8, 1.2) if avg_rep >= 1.0 else ((1.2, 2.8) if avg_rep <= 0.5 else (2.0, 1.8))
        
        last_s = max(d['sorteo'] for d in train_draws)
        last_d = [d for d in train_draws if d['sorteo'] == last_s]
        res_c = Counter(n for d in last_d for n in d['numbers'])
        prev_ss = set(n for d in last_d if d['modalidad'] == 'SiempreSale' for n in d['numbers'])
        
        saturated_decades = set()
        last_trad = trad_series[-1]
        for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
            if cnt >= 3: saturated_decades.add(dec)
            
        antag = compute_antagonist_pairs(train_draws)
        recent_numbers = set(n for d in train_draws[-5:] for n in d['numbers'])
        
        borda_base = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda_base[n] += w * (45 - rank)
            
        for n in range(46):
            if res_c[n] >= 2: borda_base[n] += 18.0 * (res_c[n] - 1)
            elif res_c[n] == 1: borda_base[n] += 4.0
            if n in prev_ss: borda_base[n] += 4.0
            
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        
        def run_pipeline_with_window(w_size, w_scale):
            c_day = Counter(n for d in same_day_draws[-w_size:] for n in d['numbers'])
            b_day = defaultdict(float)
            b_hib = defaultdict(float)
            for n in range(46):
                b_day[n] = borda_base[n] + w_scale * c_day[n]
                b_hib[n] = borda_base[n] + (w_scale / 2.0) * c_day[n]
                
            p_base = select_pool_9_elite(borda_base, saturated_decades)
            p_day = select_pool_9_elite(b_day, saturated_decades)
            p_hib = select_pool_9_elite(b_hib, saturated_decades)
            
            def get_v(p):
                c_list = list(itertools.combinations(p, 6))
                v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=3)]
                if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
                return v or c_list[:10]
                
            v_b, v_d, v_h = get_v(p_base), get_v(p_day), get_v(p_hib)
            t_b = select_optimal_wheel_triplet(v_b, p_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
            b1 = t_b[0]
            
            cands_b2 = [c for c in v_d if 2 <= len(set(c) & set(b1)) <= 4] or [c for c in v_d if c != b1]
            b2 = max(cands_b2, key=lambda c: sum(b_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
            
            q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
            cands_b3 = [c for c in v_h if 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4 and len(set(b1) & set(b2) & set(c)) <= 2]
            if not cands_b3: cands_b3 = [c for c in v_h if c != b1 and c != b2 and len(set(c) & set(b1)) <= 4 and len(set(c) & set(b2)) <= 4] or [c for c in v_h if c != b1 and c != b2]
            b3 = max(cands_b3, key=lambda c: sum(b_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12))
            return [b1, b2, b3]

        t_1 = run_pipeline_with_window(40, 16.0)
        t_2 = run_pipeline_with_window(80, 8.0)
        t_3 = run_pipeline_with_window(120, 5.3)
        
        for name, t_list in [('1. Ventana 40 (10 semanas)', t_1),
                             ('2. Ventana 80 (20 semanas)', t_2),
                             ('3. Ventana 120 (30 semanas)', t_3)]:
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
    print(f"\n[OK] Benchmark completado en {elapsed:.1f} segundos.\n")
    
    print("="*125)
    print(f"{'Estrategia Evaluada (500 Sorteos)':<65} | {'Prom. Trad.':<10} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*125)
    for name, s in metrics.items():
        avg_t = np.mean(s['max_trad'])
        p3 = (s['trad_3+'] / n_eval) * 100
        n4_t = s['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = s['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = s['pe_5+']
        print(f"{name:<65} | {avg_t:<10.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*125)

if __name__ == '__main__':
    run_day_window_test(500)

