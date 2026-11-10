"""
Test de Combinaciones: Opcion A (Tradicional puro) + Opcion B (Dia de la Semana)
Evaluando 500 sorteos (#2923 a #3422)
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
from cobertura_combinatoria import select_optimal_wheel_triplet

# Fast Multi-Output Attention MLP
def compute_fast_attention_mlp_scores(train_draws):
    trad_draws = [d['numbers'] for d in train_draws if d['modalidad'] == 'Tradicional'][-120:]
    T = len(trad_draws)
    if T < 15:
        return np.ones(46) * 0.05
    matrix = np.zeros((T, 46))
    for t, nums in enumerate(trad_draws):
        for n in nums: matrix[t, n] = 1.0
        
    X_list, y_list = [], []
    for t in range(10, T):
        feat = np.hstack([matrix[t-1], matrix[t-2], matrix[t-3], matrix[t-4], matrix[t-5], matrix[t-8], matrix[t-10]])
        X_list.append(feat)
        y_list.append(matrix[t])
        
    X = np.array(X_list)
    Y = np.array(y_list)
    last_feat = np.hstack([matrix[-1], matrix[-2], matrix[-3], matrix[-4], matrix[-5], matrix[-8], matrix[-10]]).reshape(1, -1)
    
    try:
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, random_state=42, early_stopping=True)
        mlp.fit(X, Y)
        scores = mlp.predict_proba(last_feat)[0]
    except Exception:
        scores = np.mean(matrix[-20:], axis=0)
    return scores

def run_hybrid_benchmark():
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-500:]
    n_eval = len(eval_draws)
    
    print("="*110)
    print(f" EVALUACION HIBRIDA: OPCION A + OPCION B EN {n_eval} SORTEOS")
    print("="*110)
    
    strategies = {
        '1. Opcion A Pura (3 Boletos Tradicional Puro)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'boletos_4': 0, 'max_trad': []
        },
        '2. Opcion B Pura (3 Boletos Dia de Semana)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'boletos_4': 0, 'max_trad': []
        },
        '3. Hibrido Calibrado (1 solo Pool con w_day=8.0)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'boletos_4': 0, 'max_trad': []
        },
        '4. Portfolio Especializado (3 Boletos: 1 Trad + 1 Dia + 1 Consenso)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'boletos_4': 0, 'max_trad': []
        },
        '5. Cartera Dual Completa (6 Boletos: 3 de A + 3 de B)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'boletos_4': 0, 'max_trad': []
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
        
        # 1. Extraccion base
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
        
        # Regimen Dinamico
        last_5 = train_draws[-5:]
        n_repeats = sum(len(set(last_5[i]['numbers']) & set(last_5[i-1]['numbers'])) for i in range(1, len(last_5)))
        avg_repeats = n_repeats / 4.0
        if avg_repeats >= 1.25:
            w_rec, w_ddm = 3.0, 1.0
        elif avg_repeats <= 0.75:
            w_rec, w_ddm = 1.0, 3.0
        else:
            w_rec, w_ddm = 2.0, 1.8
            
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        # Borda Base (Opcion A)
        borda_base = defaultdict(float)
        models_base = [
            (counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm),
            ({n: attn_scores[n] for n in range(46)}, 1.5)
        ]
        for m_scores, w in models_base:
            ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
            for rank, n in enumerate(ranked):
                borda_base[n] += w * (45 - rank)
                
        ranking_base = sorted(range(46), key=lambda x: borda_base[x], reverse=True)
        pool_9_base = sorted(ranking_base[:9])
        combos_9_base = list(itertools.combinations(pool_9_base, 6))
        viable_base = [c for c in combos_9_base if es_ticket_valido(c, recent_numbers)] or combos_9_base
        t_base = select_optimal_wheel_triplet(viable_base, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # Borda Dia de Semana (Opcion B)
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
        borda_day = defaultdict(float)
        for n in range(46):
            borda_day[n] = borda_base[n] + 16.0 * counts_day[n]
        ranking_day = sorted(range(46), key=lambda x: borda_day[x], reverse=True)
        pool_9_day = sorted(ranking_day[:9])
        combos_day = list(itertools.combinations(pool_9_day, 6))
        viable_day = [c for c in combos_day if es_ticket_valido(c, recent_numbers)] or combos_day
        t_day = select_optimal_wheel_triplet(viable_day, pool_9_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # Hibrido Calibrado (Ensamble w_day = 8.0)
        borda_hib = defaultdict(float)
        for n in range(46):
            borda_hib[n] = borda_base[n] + 8.0 * counts_day[n]
        ranking_hib = sorted(range(46), key=lambda x: borda_hib[x], reverse=True)
        pool_9_hib = sorted(ranking_hib[:9])
        combos_hib = list(itertools.combinations(pool_9_hib, 6))
        viable_hib = [c for c in combos_hib if es_ticket_valido(c, recent_numbers)] or combos_hib
        t_hib = select_optimal_wheel_triplet(viable_hib, pool_9_hib, borda_hib, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # Portfolio Especializado de 3 Boletos:
        # Boleto 1: El mejor boleto de Opcion A (Tradicional)
        # Boleto 2: El mejor boleto de Opcion B (Dia de Semana)
        # Boleto 3: El mejor boleto de consenso/hibrido
        b1 = t_base[0]
        b2 = t_day[0] if t_day[0] != b1 else t_day[1]
        b3 = t_hib[0] if (t_hib[0] != b1 and t_hib[0] != b2) else t_base[1]
        t_portfolio = [b1, b2, b3]
        
        # Cartera Dual Completa (6 Boletos)
        t_dual_6 = t_base + t_day
        
        candidates = [
            ('1. Opcion A Pura (3 Boletos Tradicional Puro)', t_base),
            ('2. Opcion B Pura (3 Boletos Dia de Semana)', t_day),
            ('3. Hibrido Calibrado (1 solo Pool con w_day=8.0)', t_hib),
            ('4. Portfolio Especializado (3 Boletos: 1 Trad + 1 Dia + 1 Consenso)', t_portfolio),
            ('5. Cartera Dual Completa (6 Boletos: 3 de A + 3 de B)', t_dual_6)
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
    print(f"\n[OK] Simulacion completada en {elapsed:.1f} segundos.\n")
    
    print("="*118)
    print(f"{'Estrategia / Arquitectura Evaluada':<65} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*118)
    for name, s in strategies.items():
        avg_t = np.mean(s['max_trad'])
        p3 = (s['trad_3+'] / n_eval) * 100
        n4_t = s['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = s['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = s['pe_5+']
        print(f"{name:<65} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*118)

if __name__ == '__main__':
    run_hybrid_benchmark()
