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
    load_data, es_ticket_valido, compute_adaptive_pool,
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

# 1. Validación Idea 1: Filtro de Terminaciones (máximo 2 con mismo último dígito)
def es_ticket_terminaciones(ticket, recent_numbers=None):
    if not es_ticket_valido(ticket, recent_numbers):
        return False
    endings = Counter(n % 10 for n in ticket)
    if max(endings.values()) > 2:
        return False
    return True

# 4. Cálculo de Ventana de Maduración (Idea 4)
def compute_maturation_scores(train_draws):
    # Atraso actual para cada número
    last_seen = {n: -1 for n in range(46)}
    for idx, d in enumerate(train_draws):
        for n in d['numbers']:
            last_seen[n] = idx
    n_total = len(train_draws)
    mat_scores = np.zeros(46)
    for n in range(46):
        delay = n_total - 1 - last_seen[n] if last_seen[n] != -1 else n_total
        # Campana gaussiana centrada en retraso 7.5, sigma 3.5 (cubre ventana 4 a 11)
        # y penaliza sequías profundas (>16)
        mat_scores[n] = np.exp(-((delay - 7.5) ** 2) / (2 * (3.5 ** 2)))
    return mat_scores

def run_benchmark_new_ideas():
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-500:]
    n_eval = len(eval_draws)
    
    print("="*105)
    print(f" BENCHMARK DE 4 NUEVAS IDEAS EN {n_eval} SORTEOS HISTÓRICOS (CASI 5 AÑOS DE HISTORIA)")
    print(f" Periodo: Sorteo #{eval_draws[0]['sorteo']} ({eval_draws[0]['fecha']}) al #{eval_draws[-1]['sorteo']} ({eval_draws[-1]['fecha']})")
    print(" Evaluando: Baseline BIBD Dinámico vs Terminaciones vs Día de Semana vs Pares Antagónicos vs Ventana Maduración vs Combo")
    print("="*105)
    
    strategies = {
        '0. Baseline Actual (BIBD Dinámico Pool 9)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_4': 0
        },
        '1. Idea 1: Filtro de Terminaciones (Max 2 por último dígito)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_4': 0
        },
        '2. Idea 2: Modelado por Día de Sorteo (Miércoles vs Domingo)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_4': 0
        },
        '3. Idea 3: Filtro de Pares Antagónicos (Lift < 0.35)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_4': 0
        },
        '4. Idea 4: Ventana de Maduración de Atraso (Curva 4-11)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_4': 0
        },
        '5. Combo All-Star (Terminaciones + Maduración + Día)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_4': 0
        }
    }
    
    start_time = time.time()
    
    for step_i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        target_fecha = target_draw['fecha']
        target_weekday = datetime.strptime(target_fecha, '%Y-%m-%d').weekday() # 2=Miercoles, 6=Domingo
        
        group = [d for d in draws if d['sorteo'] == target_sorteo]
        real_by_mod = {d['modalidad']: set(d['numbers']) for d in group}
        target_trad = real_by_mod.get('Tradicional', set())
        target_seg = real_by_mod.get('Segunda', set())
        target_rev = real_by_mod.get('Revancha', set())
        target_ss = real_by_mod.get('SiempreSale', set())
        pozo_extra = target_trad | target_seg | target_rev
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        
        # 1. Extracción base de características
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
        
        # Detección de régimen dinámico base
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
        
        # Borda Base Dinámica
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
        
        # -------------------------------------------------------------
        # 0. BASELINE ACTUAL
        # -------------------------------------------------------------
        t_base = select_optimal_wheel_triplet(viable_base, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # -------------------------------------------------------------
        # 1. IDEA 1: FILTRO DE TERMINACIONES
        # -------------------------------------------------------------
        viable_term = [c for c in combos_9_base if es_ticket_terminaciones(c, recent_numbers)]
        if len(viable_term) < 3: viable_term = viable_base
        t_term = select_optimal_wheel_triplet(viable_term, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # -------------------------------------------------------------
        # 2. IDEA 2: MODELADO POR DÍA DE SEMANA
        # Ponderar con frecuencia específica del mismo día (Miércoles vs Domingo)
        # -------------------------------------------------------------
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
        
        # -------------------------------------------------------------
        # 3. IDEA 3: FILTRO DE PARES ANTAGÓNICOS (Lift < 0.35)
        # -------------------------------------------------------------
        # Calcular pares antagónicos con lift bajo
        T_hist = len(train_draws)
        freq_hist = Counter(n for d in train_draws for n in d['numbers'])
        pair_cnt = Counter(tuple(sorted([u, v])) for d in train_draws for u, v in itertools.combinations(d['numbers'], 2))
        antagonist_pairs = set()
        for u, v in itertools.combinations(range(46), 2):
            p_exp = (freq_hist[u] / T_hist) * (freq_hist[v] / T_hist) * T_hist
            if p_exp >= 7.0 and pair_cnt[tuple(sorted([u, v]))] <= 2:
                antagonist_pairs.add(tuple(sorted([u, v])))
                
        def es_ticket_sin_antagonistas(ticket):
            for u, v in itertools.combinations(ticket, 2):
                if tuple(sorted([u, v])) in antagonist_pairs:
                    return False
            return True
            
        viable_antag = [c for c in viable_base if es_ticket_sin_antagonistas(c)]
        if len(viable_antag) < 3: viable_antag = viable_base
        t_antag = select_optimal_wheel_triplet(viable_antag, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # -------------------------------------------------------------
        # 4. IDEA 4: VENTANA DE MADURACIÓN DE ATRASO (Curva 4 a 11)
        # -------------------------------------------------------------
        mat_scores = compute_maturation_scores(train_draws)
        borda_mat = defaultdict(float)
        for n in range(46):
            borda_mat[n] = borda_base[n] + 45.0 * mat_scores[n]
        ranking_mat = sorted(range(46), key=lambda x: borda_mat[x], reverse=True)
        pool_9_mat = sorted(ranking_mat[:9])
        combos_mat = list(itertools.combinations(pool_9_mat, 6))
        viable_mat = [c for c in combos_mat if es_ticket_valido(c, recent_numbers)] or combos_mat
        t_mat = select_optimal_wheel_triplet(viable_mat, pool_9_mat, borda_mat, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # -------------------------------------------------------------
        # 5. COMBO ALL-STAR (Terminaciones + Maduración + Día)
        # -------------------------------------------------------------
        borda_combo = defaultdict(float)
        for n in range(46):
            borda_combo[n] = borda_base[n] + 12.0 * counts_day[n] + 35.0 * mat_scores[n]
        ranking_combo = sorted(range(46), key=lambda x: borda_combo[x], reverse=True)
        pool_9_combo = sorted(ranking_combo[:9])
        combos_combo = list(itertools.combinations(pool_9_combo, 6))
        viable_combo = [c for c in combos_combo if es_ticket_terminaciones(c, recent_numbers)]
        if len(viable_combo) < 3: viable_combo = combos_combo
        t_combo = select_optimal_wheel_triplet(viable_combo, pool_9_combo, borda_combo, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        candidates = [
            ('0. Baseline Actual (BIBD Dinámico Pool 9)', t_base, pool_9_base),
            ('1. Idea 1: Filtro de Terminaciones (Max 2 por último dígito)', t_term, pool_9_base),
            ('2. Idea 2: Modelado por Día de Sorteo (Miércoles vs Domingo)', t_day, pool_9_day),
            ('3. Idea 3: Filtro de Pares Antagónicos (Lift < 0.35)', t_antag, pool_9_base),
            ('4. Idea 4: Ventana de Maduración de Atraso (Curva 4-11)', t_mat, pool_9_mat),
            ('5. Combo All-Star (Terminaciones + Maduración + Día)', t_combo, pool_9_combo)
        ]
        
        for name, t_list, p_list in candidates:
            p_hits = len(set(p_list) & target_trad) if p_list else 0
            hits_trad = [len(set(t) & target_trad) for t in t_list]
            max_h_trad = max(hits_trad)
            max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
            max_pe = max(len(set(t) & pozo_extra) for t in t_list)
            
            if p_hits >= 3 and max_h_trad < 3:
                strategies[name]['desparramados'] += 1
                
            strategies[name]['max_trad'].append(max_h_trad)
            strategies[name]['boletos_4'] += sum(1 for h in hits_trad if h >= 4)
            
            if max_h_trad >= 3: strategies[name]['trad_3+'] += 1
            if max_h_trad >= 4: strategies[name]['trad_4+'] += 1
            if max_any >= 4: strategies[name]['any_4+'] += 1
            if max_pe >= 5: strategies[name]['pe_5+'] += 1
            if max_pe >= 6: strategies[name]['pe_6'] += 1
            
        if (step_i + 1) % 100 == 0:
            print(f" -> Progreso: {step_i + 1} de {n_eval} sorteos completados...")

    elapsed = time.time() - start_time
    print(f"\n[OK] Simulación de 500 sorteos finalizada en {elapsed:.1f} segundos.\n")
    
    print("="*112)
    print(f"{'Estrategia / Idea Evaluada (500 Sorteos)':<58} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*112)
    for name, s in strategies.items():
        avg_t = np.mean(s['max_trad'])
        p3 = (s['trad_3+'] / n_eval) * 100
        n4_t = s['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = s['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = s['pe_5+']
        pe6 = s['pe_6']
        desp = s['desparramados']
        print(f"{name:<58} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*112)
    
    print("\n" + "="*85)
    print(" DETALLES CRÍTICOS (500 SORTEOS):")
    print("="*85)
    for name, s in strategies.items():
        print(f" * {name}:")
        print(f"    - Boletos con 4+ aciertos: {s['boletos_4']} boletos")
        print(f"    - Sorteos desparramados: {s['desparramados']} fechas")
        print(f"    - Pozo Extra (5+ aciertos): {s['pe_5+']} fechas")
    print("="*85)

if __name__ == '__main__':
    run_benchmark_new_ideas()

