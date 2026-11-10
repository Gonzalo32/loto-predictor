"""
Verificación de Robustez: 600 Sorteos Consecutivos (2020-2026)
Estrategia 1 (Producción Actual) vs Estrategia 3 (Arrastre Global Multi-Modal Suave)
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

def run_600_test(n_eval=600):
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]
    
    print("="*115)
    print(f" BENCHMARK DEFINITIVO 600 SORTEOS (6 AÑOS): PRODUCCIÓN ACTUAL vs ARRASTRE MULTI-MODAL SUAVE")
    print(f" Periodo: Sorteo #{eval_draws[0]['sorteo']} al #{eval_draws[-1]['sorteo']}")
    print("="*115)
    
    metrics = {
        '1. Producción Actual': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'boletos_4': 0
        },
        '2. Con Arrastre Multi-Modal Suave (Prev Date +4, SS +6)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'boletos_4': 0
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
        
        trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
        last_5_trad = trad_series[-5:]
        n_rep = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
        avg_rep = n_rep / 4.0
        if avg_rep >= 1.0: w_rec, w_ddm = 2.8, 1.2
        elif avg_rep <= 0.5: w_rec, w_ddm = 1.2, 2.8
        else: w_rec, w_ddm = 2.0, 1.8
        
        last_sorteo_num = max(d['sorteo'] for d in train_draws)
        last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
        res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
        
        prev_ss_nums = set()
        for d in last_date_draws:
            if d['modalidad'] == 'SiempreSale':
                prev_ss_nums.update(d['numbers'])
                
        saturated_decades = set()
        last_trad = trad_series[-1]
        for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
            if cnt >= 3: saturated_decades.add(dec)
            
        antag = compute_antagonist_pairs(train_draws)
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
        
        borda_base_calc = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda_base_calc[n] += w * (45 - rank)
            
        for n in range(46):
            if res_counts[n] >= 2:
                borda_base_calc[n] += 18.0 * (res_counts[n] - 1)
                
        def evaluate_portfolio(borda_dict):
            b_day = defaultdict(float)
            b_hib = defaultdict(float)
            for n in range(46):
                b_day[n] = borda_dict[n] + 16.0 * counts_day[n]
                b_hib[n] = borda_dict[n] + 8.0 * counts_day[n]
                
            p_base = select_pool_9_elite(borda_dict, saturated_decades)
            p_day = select_pool_9_elite(b_day, saturated_decades)
            p_hib = select_pool_9_elite(b_hib, saturated_decades)
            
            def get_v(p):
                c_list = list(itertools.combinations(p, 6))
                v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)]
                if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=4)]
                if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
                return v or c_list[:10]
                
            t_b = select_optimal_wheel_triplet(get_v(p_base), p_base, borda_dict, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
            t_d = select_optimal_wheel_triplet(get_v(p_day), p_day, b_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
            t_h = select_optimal_wheel_triplet(get_v(p_hib), p_hib, b_hib, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
            
            b1 = t_b[0]
            b2 = t_d[0] if t_d[0] != b1 else t_d[1]
            q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
            best_b3 = None
            best_score = -float('inf')
            for c in get_v(p_hib):
                if c == b1 or c == b2: continue
                q_c = set(itertools.combinations(c, 4))
                score_c = sum(b_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(q_c - q12)
                if score_c > best_score:
                    best_score = score_c
                    best_b3 = c
            b3 = best_b3 if best_b3 is not None else (t_h[0] if (t_h[0] != b1 and t_h[0] != b2) else t_b[1])
            return [b1, b2, b3]

        # 1. Producción Actual
        borda_1 = borda_base_calc.copy()
        t_1 = evaluate_portfolio(borda_1)
        
        # 2. Arrastre Suave
        borda_2 = borda_base_calc.copy()
        for n in range(46):
            if res_counts[n] == 1:
                borda_2[n] += 4.0
            if n in prev_ss_nums:
                borda_2[n] += 4.0
        t_2 = evaluate_portfolio(borda_2)
        
        for name, t_list in [('1. Producción Actual', t_1),
                             ('2. Con Arrastre Multi-Modal Suave (Prev Date +4, SS +6)', t_2)]:
            hits_trad = [len(set(t) & target_trad) for t in t_list]
            max_h_trad = max(hits_trad)
            max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
            max_pe = max(len(set(t) & pozo_extra) for t in t_list)
            
            metrics[name]['max_trad'].append(max_h_trad)
            metrics[name]['boletos_4'] += sum(1 for h in hits_trad if h >= 4)
            if max_h_trad >= 3: metrics[name]['trad_3+'] += 1
            if max_h_trad >= 4: metrics[name]['trad_4+'] += 1
            if max_any >= 4: metrics[name]['any_4+'] += 1
            if max_pe >= 5: metrics[name]['pe_5+'] += 1
            
        if (step_i + 1) % 100 == 0:
            print(f" -> Progreso: {step_i + 1} de {n_eval} sorteos completados...")
            
    elapsed = time.time() - start_time
    print(f"\n[OK] Benchmark de {n_eval} sorteos finalizado en {elapsed:.1f} segundos.\n")
    
    print("="*125)
    print(f"{'Estrategia Evaluada (600 Sorteos)':<60} | {'Prom. Trad.':<10} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*125)
    for name, s in metrics.items():
        avg_t = np.mean(s['max_trad'])
        p3 = (s['trad_3+'] / n_eval) * 100
        n4_t = s['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = s['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = s['pe_5+']
        print(f"{name:<60} | {avg_t:<10.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*125)

if __name__ == '__main__':
    run_600_test(600)

