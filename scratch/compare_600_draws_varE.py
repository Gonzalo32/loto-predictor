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

def select_pool(borda_scores, saturated_decades=None, max_per_decade=3, pool_size=9):
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
            if len(pool) == pool_size: break
    if len(pool) < pool_size:
        for n in ranked:
            if n not in pool:
                pool.append(n)
                if len(pool) == pool_size: break
    return sorted(pool)

def es_ticket_valido_enhanced(ticket, recent_numbers=None, saturated_decades=None, antagonist_pairs=None, 
                              min_distinct_decades=3, filter_sum=True, filter_hilow=True):
    if not es_ticket_valido(ticket, recent_numbers, saturated_decades, antagonist_pairs, min_distinct_decades):
        return False
    if filter_sum:
        s = sum(ticket)
        if s < 68 or s > 202:
            return False
    if filter_hilow:
        lows = sum(1 for n in ticket if n <= 22)
        if lows < 1 or lows > 5:
            return False
    return True

def run_600_draw_comparison():
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    n_eval = 600
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]
    
    configs = {
        'Baseline (Producción Actual)': {
            'pool3_size': 9, 'filter_sum': False, 'filter_hilow': False
        },
        'Sistema Mejorado (Var E: MacroFiltros + Pool 10 B3)': {
            'pool3_size': 10, 'filter_sum': True, 'filter_hilow': True
        }
    }
    
    results = {cfg_name: {
        'trad_hits': [], 'trad_3': 0, 'trad_4': 0, 'trad_5': 0, 'trad_6': 0,
        'seg_hits': [],  'seg_3': 0,  'seg_4': 0,  'seg_5': 0,
        'rev_hits': [],  'rev_3': 0,  'rev_4': 0,
        'ss_hits': [],   'ss_3': 0,   'ss_4': 0,
        'global_4plus': 0, 'global_3plus': 0, 'pe_4plus': 0, 'pe_5plus': 0, 'pe_6': 0,
        'overlaps_ge_5': 0
    } for cfg_name in configs}
    
    t0 = time.time()
    print(f"Iniciando Backtest de 600 Sorteos (6 años completos: 2021-2026)...")
    
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
        
        # Precomputes comunes
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
        if avg_rep >= 1.0: w_rec, w_ddm = 2.8, 1.2
        elif avg_rep <= 0.5: w_rec, w_ddm = 1.2, 2.8
        else: w_rec, w_ddm = 2.0, 1.8
        
        last_sorteo_num = max(d['sorteo'] for d in train_draws)
        last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
        res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
        prev_ss_nums = set(n for d in last_date_draws if d['modalidad'] == 'SiempreSale' for n in d['numbers'])
        
        saturated_decades = set()
        last_trad = trad_series[-1]
        for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
            if cnt >= 3: saturated_decades.add(dec)
            
        antag = compute_antagonist_pairs(train_draws)
        recent_numbers = set(n for d in train_draws[-5:] for n in d['numbers'])
        
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
        
        borda = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
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
        
        for cfg_name, cfg in configs.items():
            p_hib = select_pool(borda_hib, saturated_decades, pool_size=cfg['pool3_size'])
            
            def get_v(p):
                c_list = list(itertools.combinations(p, 6))
                v = [c for c in c_list if es_ticket_valido_enhanced(
                    c, recent_numbers, saturated_decades, antag, min_distinct_decades=3,
                    filter_sum=cfg['filter_sum'], filter_hilow=cfg['filter_hilow']
                )]
                if len(v) < 3:
                    v = [c for c in c_list if es_ticket_valido_enhanced(
                        c, recent_numbers, saturated_decades, None, min_distinct_decades=3,
                        filter_sum=cfg['filter_sum'], filter_hilow=cfg['filter_hilow']
                    )]
                if len(v) < 3:
                    v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, None, min_distinct_decades=3)]
                return v or c_list[:10]
                
            v_base = get_v(p_base)
            v_day = get_v(p_day)
            v_hib = get_v(p_hib)
            
            t_b = select_optimal_wheel_triplet(v_base, p_base, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
            b1 = t_b[0]
            
            cands_b2 = [c for c in v_day if 2 <= len(set(c) & set(b1)) <= 4]
            if not cands_b2: cands_b2 = [c for c in v_day if c != b1]
            b2 = max(cands_b2, key=lambda c: sum(borda_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
            
            q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
            cands_b3 = [
                c for c in v_hib
                if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2
            ]
            if not cands_b3: cands_b3 = [c for c in v_hib if c != b1 and c != b2 and len(set(c) & set(b1)) <= 4 and len(set(c) & set(b2)) <= 4]
            if not cands_b3: cands_b3 = [c for c in v_hib if c != b1 and c != b2]
            
            b3 = max(
                cands_b3,
                key=lambda c: sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12)
            )
            
            tickets = [b1, b2, b3]
            
            # Check overlap integrity
            o12 = len(set(b1) & set(b2))
            o13 = len(set(b1) & set(b3))
            o23 = len(set(b2) & set(b3))
            if max(o12, o13, o23) >= 5:
                results[cfg_name]['overlaps_ge_5'] += 1
            
            h_trad = max(len(set(t) & target_trad) for t in tickets)
            h_seg = max(len(set(t) & target_seg) for t in tickets) if target_seg else 0
            h_rev = max(len(set(t) & target_rev) for t in tickets) if target_rev else 0
            h_ss  = max(len(set(t) & target_ss) for t in tickets) if target_ss else 0
            h_pe  = max(len(set(t) & pozo_extra) for t in tickets)
            
            res = results[cfg_name]
            res['trad_hits'].append(h_trad)
            if h_trad >= 3: res['trad_3'] += 1
            if h_trad >= 4: res['trad_4'] += 1
            if h_trad >= 5: res['trad_5'] += 1
            if h_trad == 6: res['trad_6'] += 1
            
            if target_seg:
                res['seg_hits'].append(h_seg)
                if h_seg >= 3: res['seg_3'] += 1
                if h_seg >= 4: res['seg_4'] += 1
                if h_seg >= 5: res['seg_5'] += 1
                
            if target_rev:
                res['rev_hits'].append(h_rev)
                if h_rev >= 3: res['rev_3'] += 1
                if h_rev >= 4: res['rev_4'] += 1
                
            if target_ss:
                res['ss_hits'].append(h_ss)
                if h_ss >= 3: res['ss_3'] += 1
                if h_ss >= 4: res['ss_4'] += 1
                
            if max(h_trad, h_seg, h_rev, h_ss) >= 4:
                res['global_4plus'] += 1
            if max(h_trad, h_seg, h_rev, h_ss) >= 3:
                res['global_3plus'] += 1
            if h_pe >= 4: res['pe_4plus'] += 1
            if h_pe >= 5: res['pe_5plus'] += 1
            if h_pe == 6: res['pe_6'] += 1

        if (step_i + 1) % 100 == 0:
            print(f"[{step_i + 1}/{n_eval}] completados en {time.time()-t0:.1f}s...")
            
    print("\n" + "="*120)
    print(f" RESULTADOS COMPARATIVOS OFICIALES: {n_eval} SORTEOS (2021-2026)")
    print("="*120)
    for cfg_name, res in results.items():
        print(f"\n>>> {cfg_name.upper()}:")
        print(f"  * Tradicional 3+ aciertos : {res['trad_3']:>2} fechas ({res['trad_3']/n_eval*100:>4.2f}%) | Promedio aciertos: {np.mean(res['trad_hits']):.2f}")
        print(f"  * Tradicional 4+ (PREMIO) : {res['trad_4']:>2} fechas ({res['trad_4']/n_eval*100:>4.2f}%)")
        print(f"  * Tradicional 5+          : {res['trad_5']:>2} fechas")
        print(f"  * Segunda Vuelta 4+       : {res['seg_4']:>2} fechas")
        print(f"  * Revancha 4+             : {res['rev_4']:>2} fechas")
        print(f"  * Siempre Sale 4+         : {res['ss_4']:>2} fechas")
        print(f"  * GLOBAL CUPÓN 4+ (PREMIO): {res['global_4plus']:>2} fechas ({res['global_4plus']/n_eval*100:>4.2f}%)")
        print(f"  * GLOBAL CUPÓN 3+ (RETORNO): {res['global_3plus']:>2} fechas ({res['global_3plus']/n_eval*100:>4.2f}%)")
        print(f"  * Pozo Extra 4+ aciertos  : {res['pe_4plus']:>2} fechas ({res['pe_4plus']/n_eval*100:>4.2f}%)")
        print(f"  * Pozo Extra 5+ aciertos  : {res['pe_5plus']:>2} fechas ({res['pe_5plus']/n_eval*100:>4.2f}%)")
        print(f"  * Boletos >= 5 solapados  : {res['overlaps_ge_5']} (casi nulo)")
    print("="*120)

if __name__ == '__main__':
    run_600_draw_comparison()

