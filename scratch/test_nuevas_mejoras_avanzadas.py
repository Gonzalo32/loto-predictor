"""
Test de 4 Nuevas Ideas Avanzadas para Maximizar Probabilidad con 3 Boletos en 500 Sorteos:
1. Maximizador Exacto de Cobertura de Cuadruplas (Wheel C(9, 6, 4))
2. Reduccion a Pool 8 (Concentracion Ultra-alta vs Pool 9)
3. Filtro de Distribucion de Decenas (Minimo 4 decenas distintas por boleto)
4. Sinergia de Pares Fuertes (Graph Clique: exigir que cada boleto contenga al menos 2 pares de alta afinidad historica)
"""
import os, sys, itertools, csv
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import time
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores
)
from scratch.test_hybrid_option_a_and_b import compute_fast_attention_mlp_scores

def es_ticket_4_decenas(ticket, recent_numbers):
    if not es_ticket_valido(ticket, recent_numbers):
        return False
    decenas = set(n // 10 for n in ticket)
    if len(decenas) < 4: # exigir dispersion en al menos 4 decenas (evita amontonamientos)
        return False
    return True

def select_max_quad_wheel(viable_combos, borda_scores, cooccur_mat):
    """
    Selecciona 3 boletos que maximicen las cuadruplas C(n, 4) unicas cubiertas
    ponderadas por el score Borda y coocurrencias.
    """
    if len(viable_combos) <= 3:
        while len(viable_combos) < 3: viable_combos.append(viable_combos[0])
        return viable_combos[:3]
    
    # Precomputar scores de cada combo
    combo_scores = []
    combo_quads = []
    for c in viable_combos:
        b_sum = sum(borda_scores[n] for n in c)
        co_sum = sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
        combo_scores.append(b_sum + 300.0 * co_sum)
        combo_quads.append(set(itertools.combinations(c, 4)))
        
    top_indices = np.argsort(combo_scores)[::-1][:25] # top 25 mejores boletos por calidad
    
    best_triplet = None
    best_obj = -1
    
    for i1, i2, i3 in itertools.combinations(top_indices, 3):
        unique_quads = len(combo_quads[i1] | combo_quads[i2] | combo_quads[i3]) # maximo 45
        quality = combo_scores[i1] + combo_scores[i2] + combo_scores[i3]
        obj = unique_quads * 100.0 + quality
        if obj > best_obj:
            best_obj = obj
            best_triplet = [viable_combos[i1], viable_combos[i2], viable_combos[i3]]
            
    return best_triplet or viable_combos[:3]

def run_advanced_ideas_benchmark():
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-500:]
    n_eval = len(eval_draws)
    
    print("="*115)
    print(f" BENCHMARK DE MEJORAS AVANZADAS EN {n_eval} SORTEOS HISTORICOS")
    print("="*115)
    
    strategies = {
        'Baseline: Portfolio Especializado (3 Boletos: Trad + Dia + Consenso)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'boletos_4': 0
        },
        'Mejora 1: Maximizador de Cobertura de Cuadruplas C(9, 6, 4)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'boletos_4': 0
        },
        'Mejora 2: Pool Ultra-Concentrado de 8 Numeros (C(8, 6))': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'boletos_4': 0
        },
        'Mejora 3: Filtro Estricto de Dispersion en 4 Decenas': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': [], 'boletos_4': 0
        },
        'Mejora 4: Sinergia Cuadruplas + 4 Decenas + Dia de Semana': {
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
        
        last_5 = train_draws[-5:]
        n_repeats = sum(len(set(last_5[i]['numbers']) & set(last_5[i-1]['numbers'])) for i in range(1, len(last_5)))
        avg_repeats = n_repeats / 4.0
        if avg_repeats >= 1.25: w_rec, w_ddm = 3.0, 1.0
        elif avg_repeats <= 0.75: w_rec, w_ddm = 1.0, 3.0
        else: w_rec, w_ddm = 2.0, 1.8
            
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        borda_base = defaultdict(float)
        models_base = [
            (counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm),
            ({n: attn_scores[n] for n in range(46)}, 1.5)
        ]
        for m_scores, w in models_base:
            ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
            for rank, n in enumerate(ranked): borda_base[n] += w * (45 - rank)
                
        # Dia de Semana
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
        borda_day = defaultdict(float)
        for n in range(46):
            borda_day[n] = borda_base[n] + 16.0 * counts_day[n]
            
        ranking_base = sorted(range(46), key=lambda x: borda_base[x], reverse=True)
        ranking_day = sorted(range(46), key=lambda x: borda_day[x], reverse=True)
        
        # 1. Baseline Portfolio
        from scratch.test_hybrid_option_a_and_b import select_optimal_wheel_triplet
        pool_9_base = sorted(ranking_base[:9])
        viable_base = [c for c in itertools.combinations(pool_9_base, 6) if es_ticket_valido(c, recent_numbers)] or list(itertools.combinations(pool_9_base, 6))
        t_base = select_optimal_wheel_triplet(viable_base, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        pool_9_day = sorted(ranking_day[:9])
        viable_day = [c for c in itertools.combinations(pool_9_day, 6) if es_ticket_valido(c, recent_numbers)] or list(itertools.combinations(pool_9_day, 6))
        t_day = select_optimal_wheel_triplet(viable_day, pool_9_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        b1 = t_base[0]
        b2 = t_day[0] if t_day[0] != b1 else t_day[1]
        b3 = t_base[1] if (t_base[1] != b1 and t_base[1] != b2) else t_day[1]
        t_baseline = [b1, b2, b3]
        
        # Mejora 1: Max Quad Wheel sobre Pool 9 Day
        t_quad = select_max_quad_wheel(viable_day, borda_day, cooccur_exp_mat)
        
        # Mejora 2: Pool Ultra-Concentrado de 8 Numeros (sobre ranking day)
        pool_8_day = sorted(ranking_day[:8])
        viable_8 = [c for c in itertools.combinations(pool_8_day, 6) if es_ticket_valido(c, recent_numbers)] or list(itertools.combinations(pool_8_day, 6))
        t_pool8 = select_optimal_wheel_triplet(viable_8, pool_8_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # Mejora 3: Filtro Estricto de 4 Decenas
        viable_4dec = [c for c in itertools.combinations(pool_9_day, 6) if es_ticket_4_decenas(c, recent_numbers)]
        if len(viable_4dec) < 3: viable_4dec = viable_day
        t_4dec = select_optimal_wheel_triplet(viable_4dec, pool_9_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # Mejora 4: Sinergia Cuadruplas + 4 Decenas
        t_sinergia = select_max_quad_wheel(viable_4dec, borda_day, cooccur_exp_mat)
        
        candidates = [
            ('Baseline: Portfolio Especializado (3 Boletos: Trad + Dia + Consenso)', t_baseline),
            ('Mejora 1: Maximizador de Cobertura de Cuadruplas C(9, 6, 4)', t_quad),
            ('Mejora 2: Pool Ultra-Concentrado de 8 Numeros (C(8, 6))', t_pool8),
            ('Mejora 3: Filtro Estricto de Dispersion en 4 Decenas', t_4dec),
            ('Mejora 4: Sinergia Cuadruplas + 4 Decenas + Dia de Semana', t_sinergia)
        ]
        
        for name, t_list in candidates:
            hits_trad = [len(set(t) & target_trad) for t in t_list]
            max_h_trad = max(hits_trad)
            max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
            max_pe = max(len(set(t) & pozo_extra) for t in t_list)
            
            strategies[name]['max_trad'].append(max_h_trad)
            strategies[name]['boletos_4'] += sum(1 for h in hits_trad if h >= 4)
            if max_h_trad >= 3: strategies[name]['trad_3+'] += 1
            if max_h_trad >= 4: strategies[name]['trad_4+'] += 1
            if max_any >= 4: strategies[name]['any_4+'] += 1
            if max_pe >= 5: strategies[name]['pe_5+'] += 1
            
        if (step_i + 1) % 100 == 0:
            print(f" -> Progreso: {step_i + 1} de {n_eval} sorteos completados...")
            
    elapsed = time.time() - start_time
    print(f"\n[OK] Benchmark avanzado finalizado en {elapsed:.1f} segundos.\n")
    
    print("="*120)
    print(f"{'Estrategia Avanzada Evaluada (3 Boletos)':<65} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*120)
    for name, s in strategies.items():
        avg_t = np.mean(s['max_trad'])
        p3 = (s['trad_3+'] / n_eval) * 100
        n4_t = s['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = s['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = s['pe_5+']
        print(f"{name:<65} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*120)

if __name__ == '__main__':
    run_advanced_ideas_benchmark()

