"""
Test de las 3 Nuevas Mejoras:
1. Super-Resonancia Multi-Modal (bonus a numeros con >= 2 apariciones en el sorteo previo)
2. Excepcion de Elite en Decenas Saturadas (no censurar a candidatos top 2)
3. Regimen Dinamico Cronologico Corregido (fechas reales de Tradicional)
Evaluando 500 sorteos historicos
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
            # Excepcion de elite: si la decena contiene al top 1 o top 2 de la IA, permitir hasta 3
            is_elite_dec = any((e // 10 == dec) for e in elite_numbers)
            if not is_elite_dec:
                limit = min(limit, 2)
            else:
                limit = 3
        if dec_counts[dec] < limit:
            pool.append(n)
            dec_counts[dec] += 1
            if len(pool) == 9:
                break
    if len(pool) < 9:
        for n in ranked:
            if n not in pool:
                pool.append(n)
                if len(pool) == 9: break
    return sorted(pool)

def run_resonance_benchmark(n_eval=500):
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]
    
    print("="*115)
    print(f" BENCHMARK DE LAS 3 NUEVAS MEJORAS (SUPER-RESONANCIA + EXCEPCION DE ELITE) EN {n_eval} SORTEOS")
    print(f" Periodo: Sorteo #{eval_draws[0]['sorteo']} ({eval_draws[0]['fecha']}) al #{eval_draws[-1]['sorteo']} ({eval_draws[-1]['fecha']})")
    print("="*115)
    
    metrics = {
        '1. Sistema Produccion Actual (Sin Resonancia)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'boletos_4': 0, 'desparramados': 0
        },
        '2. Sistema Nuevo (Con Resonancia + Elite + Regimen Corregido)': {
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
        
        # 1. Base
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
        
        # Saturated Decades y Pares Antagonicos
        saturated_decades = set()
        last_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional'][-1]
        for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
            if cnt >= 3: saturated_decades.add(dec)
            
        antag = compute_antagonist_pairs(train_draws)
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        # -----------------------------------------------------------------
        # MODELO A: ACTUAL (Régimen sin corregir, sin resonancia, saturada rígida)
        # -----------------------------------------------------------------
        last_5 = train_draws[-5:]
        n_rep_old = sum(len(set(last_5[i]['numbers']) & set(last_5[i-1]['numbers'])) for i in range(1, len(last_5)))
        avg_rep_old = n_rep_old / 4.0
        if avg_rep_old >= 1.25: w_rec_a, w_ddm_a = 3.0, 1.0
        elif avg_rep_old <= 0.75: w_rec_a, w_ddm_a = 1.0, 3.0
        else: w_rec_a, w_ddm_a = 2.0, 1.8
            
        borda_a = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, w_rec_a), (pcr_scores, 1.4), (ddm_scores, w_ddm_a), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda_a[n] += w * (45 - rank)
                
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
        borda_day_a = defaultdict(float)
        borda_hib_a = defaultdict(float)
        for n in range(46):
            borda_day_a[n] = borda_a[n] + 16.0 * counts_day[n]
            borda_hib_a[n] = borda_a[n] + 8.0 * counts_day[n]
            
        pool_base_a = select_balanced_pool_9(borda_a, saturated_decades)
        pool_day_a = select_balanced_pool_9(borda_day_a, saturated_decades)
        pool_hib_a = select_balanced_pool_9(borda_hib_a, saturated_decades)
        
        def get_v(p):
            c_list = list(itertools.combinations(p, 6))
            v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)]
            if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=4)]
            if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
            return v or c_list[:10]
            
        t_b_a = select_optimal_wheel_triplet(get_v(pool_base_a), pool_base_a, borda_a, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        t_d_a = select_optimal_wheel_triplet(get_v(pool_day_a), pool_day_a, borda_day_a, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        t_h_a = select_optimal_wheel_triplet(get_v(pool_hib_a), pool_hib_a, borda_hib_a, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        b1_a = t_b_a[0]
        b2_a = t_d_a[0] if t_d_a[0] != b1_a else t_d_a[1]
        b3_a = t_h_a[0] if (t_h_a[0] != b1_a and t_h_a[0] != b2_a) else t_b_a[1]
        t_actual = [b1_a, b2_a, b3_a]
        
        # -----------------------------------------------------------------
        # MODELO B: NUEVO MEJORADO
        # (Régimen cronológico real + Resonancia Multi-Modal + Excepción de Élite)
        # -----------------------------------------------------------------
        # 1. Régimen cronológico real
        trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
        last_5_trad = trad_series[-5:]
        n_rep_b = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
        avg_rep_b = n_rep_b / 4.0
        if avg_rep_b >= 1.0: w_rec_b, w_ddm_b = 2.8, 1.2
        elif avg_rep_b <= 0.5: w_rec_b, w_ddm_b = 1.2, 2.8
        else: w_rec_b, w_ddm_b = 2.0, 1.8
            
        # 2. Resonancia Multi-Modal de la fecha previa
        last_sorteo_num = max(d['sorteo'] for d in train_draws)
        last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
        res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
        
        borda_b = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, w_rec_b), (pcr_scores, 1.4), (ddm_scores, w_ddm_b), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda_b[n] += w * (45 - rank)
            
        # Aplicar Bonus de Súper-Resonancia Multi-Modal (números que salieron en 2 o 3 modalidades de la fecha previa)
        for n in range(46):
            if res_counts[n] >= 2:
                borda_b[n] += 18.0 * (res_counts[n] - 1)
                
        borda_day_b = defaultdict(float)
        borda_hib_b = defaultdict(float)
        for n in range(46):
            borda_day_b[n] = borda_b[n] + 16.0 * counts_day[n]
            borda_hib_b[n] = borda_b[n] + 8.0 * counts_day[n]
            
        # 3. Selección con Excepción de Élite en Decenas Saturadas
        pool_base_b = select_balanced_pool_9_elite(borda_b, saturated_decades)
        pool_day_b = select_balanced_pool_9_elite(borda_day_b, saturated_decades)
        pool_hib_b = select_balanced_pool_9_elite(borda_hib_b, saturated_decades)
        
        t_b_b = select_optimal_wheel_triplet(get_v(pool_base_b), pool_base_b, borda_b, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        t_d_b = select_optimal_wheel_triplet(get_v(pool_day_b), pool_day_b, borda_day_b, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        t_h_b = select_optimal_wheel_triplet(get_v(pool_hib_b), pool_hib_b, borda_hib_b, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        b1_b = t_b_b[0]
        b2_b = t_d_b[0] if t_d_b[0] != b1_b else t_d_b[1]
        b3_b = t_h_b[0] if (t_h_b[0] != b1_b and t_h_b[0] != b2_b) else t_b_b[1]
        t_mejorado = [b1_b, b2_b, b3_b]
        
        # -----------------------------------------------------------------
        # Evaluación de ambas estrategias
        # -----------------------------------------------------------------
        for name, t_list in [('1. Sistema Produccion Actual (Sin Resonancia)', t_actual),
                             ('2. Sistema Nuevo (Con Resonancia + Elite + Regimen Corregido)', t_mejorado)]:
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
    print(f"\n[OK] Benchmark de 500 sorteos finalizado en {elapsed:.1f} segundos.\n")
    
    print("="*120)
    print(f"{'Estrategia Evaluada (500 Sorteos)':<62} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*120)
    for name, s in metrics.items():
        avg_t = np.mean(s['max_trad'])
        p3 = (s['trad_3+'] / n_eval) * 100
        n4_t = s['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = s['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = s['pe_5+']
        print(f"{name:<62} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*120)

if __name__ == '__main__':
    run_resonance_benchmark(500)

