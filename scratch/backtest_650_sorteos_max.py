"""
Backtest Historico Maximo Disponible: 650 Sorteos Consecutivos (2020-2026)
Evaluacion del Sistema Nuevo vs Sistema Viejo
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

def run_max_backtest(n_eval=650):
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]
    
    print("="*115)
    print(f" BACKTEST HISTORICO MAXIMO: {n_eval} SORTEOS CONSECUTIVOS DE QUINI 6")
    print(f" Periodo: Sorteo #{eval_draws[0]['sorteo']} ({eval_draws[0]['fecha']}) al #{eval_draws[-1]['sorteo']} ({eval_draws[-1]['fecha']})")
    print("="*115)
    
    metrics = {
        'Sistema Viejo (Pool 12 Disperso)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'boletos_4': 0, 'desparramados': 0
        },
        'Sistema Nuevo en Produccion (Portfolio Especializado)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'boletos_4': 0, 'desparramados': 0
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
        
        last_5 = train_draws[-5:]
        n_repeats = sum(len(set(last_5[i]['numbers']) & set(last_5[i-1]['numbers'])) for i in range(1, len(last_5)))
        avg_repeats = n_repeats / 4.0
        if avg_repeats >= 1.25: w_rec, w_ddm = 3.0, 1.0
        elif avg_repeats <= 0.75: w_rec, w_ddm = 1.0, 3.0
        else: w_rec, w_ddm = 2.0, 1.8
            
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        borda_estatico = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, 2.0), (pcr_scores, 1.4), (ddm_scores, 1.8), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda_estatico[n] += w * (45 - rank)
                
        borda_base = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda_base[n] += w * (45 - rank)
                
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
        borda_day = defaultdict(float)
        borda_hib = defaultdict(float)
        for n in range(46):
            borda_day[n] = borda_base[n] + 16.0 * counts_day[n]
            borda_hib[n] = borda_base[n] + 8.0 * counts_day[n]
            
        saturated_decades = set()
        last_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional'][-1]
        for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
            if cnt >= 3: saturated_decades.add(dec)
            
        antag = compute_antagonist_pairs(train_draws)
        
        # Sistema Viejo
        ranking_old = sorted(range(46), key=lambda x: borda_estatico[x], reverse=True)
        pool_12_old = sorted(ranking_old[:12])
        combos_12 = list(itertools.combinations(pool_12_old, 6))
        viable_old = [c for c in combos_12 if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)] or combos_12[:100]
        t_old = select_optimal_wheel_triplet(viable_old, pool_12_old, borda_estatico, cooccur_exp_mat, max_overlap=3, w_cooccur=200.0)
        
        # Sistema Nuevo
        pool_base = select_balanced_pool_9(borda_base, saturated_decades)
        pool_day = select_balanced_pool_9(borda_day, saturated_decades)
        pool_hib = select_balanced_pool_9(borda_hib, saturated_decades)
        
        def get_viable_nuevo(p):
            c_list = list(itertools.combinations(p, 6))
            v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)]
            if len(v) < 3:
                v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=4)]
            if len(v) < 3:
                v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
            if len(v) < 3:
                v = c_list[:10]
            return v
            
        v_base = get_viable_nuevo(pool_base)
        v_day = get_viable_nuevo(pool_day)
        v_hib = get_viable_nuevo(pool_hib)
        
        t_b = select_optimal_wheel_triplet(v_base, pool_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        t_d = select_optimal_wheel_triplet(v_day, pool_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        t_h = select_optimal_wheel_triplet(v_hib, pool_hib, borda_hib, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        b1 = t_b[0]
        b2 = t_d[0] if t_d[0] != t_b[0] else t_d[1]
        b3 = t_h[0] if (t_h[0] != b1 and t_h[0] != b2) else t_b[1]
        t_nuevo = [b1, b2, b3]
        
        for name, t_list, p_ref in [('Sistema Viejo (Pool 12 Disperso)', t_old, pool_12_old),
                                    ('Sistema Nuevo en Produccion (Portfolio Especializado)', t_nuevo, pool_base)]:
            p_hits = len(set(p_ref) & target_trad)
            hits_trad = [len(set(t) & target_trad) for t in t_list]
            max_h_trad = max(hits_trad)
            max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
            max_pe = max(len(set(t) & pozo_extra) for t in t_list)
            
            if p_hits >= 3 and max_h_trad < 3:
                metrics[name]['desparramados'] += 1
                
            metrics[name]['max_trad'].append(max_h_trad)
            metrics[name]['boletos_4'] += sum(1 for h in hits_trad if h >= 4)
            if max_h_trad >= 3: metrics[name]['trad_3+'] += 1
            if max_h_trad >= 4: metrics[name]['trad_4+'] += 1
            if max_any >= 4: metrics[name]['any_4+'] += 1
            if max_pe >= 5: metrics[name]['pe_5+'] += 1
            
        if (step_i + 1) % 100 == 0:
            print(f" -> Progreso: {step_i + 1} de {n_eval} sorteos completados...")
            
    elapsed = time.time() - start_time
    print(f"\n[OK] Backtest maximo de {n_eval} sorteos completado en {elapsed:.1f} segundos.\n")
    
    print("="*115)
    print(f"{'Sistema Evaluado (650 Sorteos Maximos)':<55} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*115)
    for name, s in metrics.items():
        avg_t = np.mean(s['max_trad'])
        p3 = (s['trad_3+'] / n_eval) * 100
        n4_t = s['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = s['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = s['pe_5+']
        print(f"{name:<55} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*115)
    
    print("\n" + "="*85)
    print(" DETALLES CRITICOS (650 SORTEOS):")
    print("="*85)
    for name, s in metrics.items():
        print(f" * {name}:")
        print(f"    - Total de premios cobrados con 4+ aciertos: {s['any_4+']} fechas ({s['boletos_4']} boletos premiados)")
        print(f"    - Fechas con 5+ aciertos en el Pozo Extra: {s['pe_5+']} fechas")
        print(f"    - Fechas desparramadas (aciertos en pool pero 0 premios): {s['desparramados']} fechas")
    print("="*85)

if __name__ == '__main__':
    run_max_backtest(650)

