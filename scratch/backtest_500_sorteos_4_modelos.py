import os, sys, itertools, csv
from collections import defaultdict, Counter
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
from cobertura_combinatoria import select_optimal_wheel_triplet, select_balanced_bibd_triplet

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

# Filtro Cuantitativo Estricto 95% CI (Idea 4)
def es_ticket_cuantil_estricto(ticket, recent_numbers=None):
    if not es_ticket_valido(ticket, recent_numbers):
        return False
    s = sum(ticket)
    if s < 90 or s > 178: return False
    spread = ticket[5] - ticket[0]
    if spread < 22 or spread > 43: return False
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares not in (2, 3, 4): return False
    return True

# Clique Core (Idea 1)
def select_bibd_clique(pool_9, borda, cooccur_matrix, boletos_viables):
    all_trios = list(itertools.combinations(pool_9, 3))
    def score_trio(trio):
        u, v, w = trio
        return (borda.get(u,0)+borda.get(v,0)+borda.get(w,0))*0.4 + (cooccur_matrix[u,v]+cooccur_matrix[u,w]+cooccur_matrix[v,w])*400.0
    core = list(max(all_trios, key=score_trio))
    rest = [n for n in pool_9 if n not in core]
    core_pairs = [(core[0], core[1]), (core[0], core[2]), (core[1], core[2])]
    pairs_of_rest = list(itertools.combinations(rest, 2))
    partitions = []
    for p1 in pairs_of_rest:
        rem1 = [n for n in rest if n not in p1]
        for p2 in itertools.combinations(rem1, 2):
            p3 = tuple(sorted([n for n in rem1 if n not in p2]))
            part = tuple(sorted([p1, p2, p3]))
            if part not in partitions: partitions.append(part)
    best_triplet = None
    best_score = -float('inf')
    viable_set = set(tuple(sorted(c)) for c in boletos_viables) if boletos_viables else None
    for part in partitions:
        for perm in itertools.permutations(part):
            t1 = tuple(sorted(list(core_pairs[0]) + [n for n in rest if n not in perm[0]]))
            if viable_set and t1 not in viable_set: continue
            t2 = tuple(sorted(list(core_pairs[1]) + [n for n in rest if n not in perm[1]]))
            if viable_set and t2 not in viable_set: continue
            t3 = tuple(sorted(list(core_pairs[2]) + [n for n in rest if n not in perm[2]]))
            if viable_set and t3 not in viable_set: continue
            total_score = sum(sum(borda.get(n, 0) for n in t) + 350.0 * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(t, 2)) for t in [t1, t2, t3])
            if total_score > best_score:
                best_score = total_score
                best_triplet = [t1, t2, t3]
    return best_triplet or select_optimal_wheel_triplet(boletos_viables or [], pool_9, borda, cooccur_matrix)

# Modelo Original (Pre-mejoras)
def old_select_triplet(boletos_viables, pool_12, scores, cooccur_matrix):
    pool_set = set(pool_12)
    scored_combos = []
    for c in boletos_viables:
        s_nodes = sum(scores.get(n, 0.0) for n in c)
        s_edges = sum(cooccur_matrix[u, v] for u, v in itertools.combinations(c, 2))
        scored_combos.append((s_nodes + 320.0 * s_edges, c))
    scored_combos.sort(key=lambda x: x[0], reverse=True)
    top_combos = scored_combos[:150]
    combo_sets = [set(c) for _, c in top_combos]
    combo_scores = [s for s, _ in top_combos]
    combo_tuples = [c for _, c in top_combos]
    combo_quartets = [set(itertools.combinations(c, 4)) for c in combo_tuples]
    best_triplet = None
    best_objective = -float('inf')
    search_limit_1 = min(25, len(top_combos))
    search_limit_2 = min(60, len(top_combos))
    for i in range(search_limit_1):
        s1 = combo_scores[i]
        c1_set = combo_sets[i]
        c1 = combo_tuples[i]
        q1 = combo_quartets[i]
        for j in range(i + 1, search_limit_2):
            c2_set = combo_sets[j]
            c2 = combo_tuples[j]
            if len(c1_set.intersection(c2_set)) > 3: continue
            covered_12 = c1_set.union(c2_set)
            uncovered = pool_set - covered_12
            if len(uncovered) > 6: continue
            s2 = combo_scores[j]
            q12 = q1.union(combo_quartets[j])
            for k in range(j + 1, min(100, len(top_combos))):
                c3_set = combo_sets[k]
                c3 = combo_tuples[k]
                if not uncovered.issubset(c3_set): continue
                if len(c1_set.intersection(c3_set)) > 3 or len(c2_set.intersection(c3_set)) > 3: continue
                if len(c1_set.intersection(c2_set).intersection(c3_set)) > 0: continue
                s3 = combo_scores[k]
                all_quads = len(q12.union(combo_quartets[k]))
                freqs = Counter(list(c1) + list(c2) + list(c3))
                score_boost = sum(scores.get(n, 0.0) * freqs[n] for n in freqs)
                obj = all_quads * 15.0 + (s1 + s2 + s3) * 0.1 + score_boost
                if obj > best_objective:
                    best_objective = obj
                    best_triplet = [c1, c2, c3]
    if best_triplet is None:
        t1 = top_combos[0][1]
        t2 = next((c for _, c in top_combos[1:] if len(set(c) & set(t1)) <= 3), top_combos[1][1])
        t3 = next((c for _, c in top_combos[2:] if len(set(c) & set(t1)) <= 3 and len(set(c) & set(t2)) <= 3), top_combos[2][1])
        best_triplet = [t1, t2, t3]
    return best_triplet

def run_500_draws_benchmark():
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    # Tomamos los últimos 500 sorteos Tradicionales
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-500:]
    n_eval = len(eval_draws)
    
    print("="*105)
    print(f" MACRO-BACKTEST WALK-FORWARD EN {n_eval} SORTEOS HISTÓRICOS (CASI 5 AÑOS DE QUINI 6)")
    print(f" Periodo: Sorteo #{eval_draws[0]['sorteo']} ({eval_draws[0]['fecha']}) al #{eval_draws[-1]['sorteo']} ({eval_draws[-1]['fecha']})")
    print(" Evaluando las 4 ideas + Baseline BIBD + Modelo Original + Azar Puro")
    print("="*105)
    
    strategies = {
        '0. Modelo Original (Pool 12, Anti-SPOF, Overlap <= 3)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'any_5+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        },
        'Baseline: BIBD Pool 9 Estático (Producción)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'any_5+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        },
        'Idea 1: Clique Core (Máxima Sinergia de Pares)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'any_5+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        },
        'Idea 2: Ensamble Dinámico por Régimen (Momentum vs Reversión)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'any_5+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        },
        'Idea 3: Especialización Multi-Modal (T1 Trad, T2 Seg, T3 SS)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'any_5+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        },
        'Idea 4: Filtros Cuantitativos 95% (Suma 90-178, Spread 22-43)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'any_5+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        },
        'Control: Azar Puro (3 Boletos Aleatorios)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'any_5+': 0, 'pe_5+': 0, 'pe_6': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        }
    }
    
    start_time = time.time()
    
    for step_i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        group = [d for d in draws if d['sorteo'] == target_sorteo]
        real_by_mod = {d['modalidad']: set(d['numbers']) for d in group}
        target_trad = real_by_mod.get('Tradicional', set())
        target_seg = real_by_mod.get('Segunda', set())
        target_rev = real_by_mod.get('Revancha', set())
        target_ss = real_by_mod.get('SiempreSale', set())
        pozo_extra = target_trad | target_seg | target_rev
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        
        # Extracción compartida de métricas para velocidad máxima
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
        
        # Borda Estándar
        borda_base = defaultdict(float)
        models_base = [
            (counts_60, 1.5), (recency_scores, 2.0), (pcr_scores, 1.4), (ddm_scores, 1.8),
            ({n: attn_scores[n] for n in range(46)}, 1.5)
        ]
        for m_scores, w in models_base:
            ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
            for rank, n in enumerate(ranked):
                borda_base[n] += w * (45 - rank)
        ranking_base = sorted(range(46), key=lambda x: borda_base[x], reverse=True)
        
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        # 0. Modelo Original (Pool 12)
        pool_12 = sorted(ranking_base[:12])
        combos_12 = [c for c in itertools.combinations(pool_12, 6) if es_ticket_valido(c, recent_numbers)] or list(itertools.combinations(pool_12, 6))[:40]
        t_orig = old_select_triplet(combos_12, pool_12, borda_base, cooccur_exp_mat)
        
        # Baseline: BIBD Pool 9
        pool_9_base = sorted(ranking_base[:9])
        combos_9_base = list(itertools.combinations(pool_9_base, 6))
        viable_9_base = [c for c in combos_9_base if es_ticket_valido(c, recent_numbers)] or combos_9_base
        t_base = select_optimal_wheel_triplet(viable_9_base, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # Idea 1: Clique Core
        t_clique = select_bibd_clique(pool_9_base, borda_base, cooccur_exp_mat, viable_9_base)
        
        # Idea 2: Ensamble Dinámico por Régimen
        last_5 = train_draws[-5:]
        n_repeats = sum(len(set(last_5[i]['numbers']) & set(last_5[i-1]['numbers'])) for i in range(1, len(last_5)))
        avg_repeats = n_repeats / 4.0
        if avg_repeats >= 1.25:
            w_rec, w_ddm = 3.0, 1.0
        elif avg_repeats <= 0.75:
            w_rec, w_ddm = 1.0, 3.0
        else:
            w_rec, w_ddm = 2.0, 1.8
            
        borda_dyn = defaultdict(float)
        models_dyn = [
            (counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm),
            ({n: attn_scores[n] for n in range(46)}, 1.5)
        ]
        for m_scores, w in models_dyn:
            ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
            for rank, n in enumerate(ranked):
                borda_dyn[n] += w * (45 - rank)
        ranking_dyn = sorted(range(46), key=lambda x: borda_dyn[x], reverse=True)
        pool_9_dyn = sorted(ranking_dyn[:9])
        combos_9_dyn = list(itertools.combinations(pool_9_dyn, 6))
        viable_9_dyn = [c for c in combos_9_dyn if es_ticket_valido(c, recent_numbers)] or combos_9_dyn
        t_dyn = select_optimal_wheel_triplet(viable_9_dyn, pool_9_dyn, borda_dyn, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # Idea 3: Especialización Multi-Modal
        draws_seg  = [d for d in train_draws if d['modalidad'] == 'Segunda']
        draws_ss   = [d for d in train_draws if d['modalidad'] == 'SiempreSale']
        counts_seg = Counter(n for d in draws_seg[-30:] for n in d['numbers'])
        counts_ss  = Counter(n for d in draws_ss[-30:] for n in d['numbers'])
        
        t1_mm = max(viable_9_base, key=lambda c: sum(borda_base[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        pool_seg = sorted(range(46), key=lambda n: borda_base[n] + 18.0 * counts_seg[n], reverse=True)[:9]
        c_seg = list(itertools.combinations(pool_seg, 6))
        v_seg = [c for c in c_seg if es_ticket_valido(c, recent_numbers)] or c_seg
        t2_mm = max(v_seg, key=lambda c: sum(borda_base[n] + 18.0 * counts_seg[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        pool_ss = sorted(range(46), key=lambda n: borda_base[n] + 18.0 * counts_ss[n], reverse=True)[:9]
        c_ss = list(itertools.combinations(pool_ss, 6))
        v_ss = [c for c in c_ss if es_ticket_valido(c, recent_numbers)] or c_ss
        t3_mm = max(v_ss, key=lambda c: sum(borda_base[n] + 18.0 * counts_ss[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        t_mm = [t1_mm, t2_mm, t3_mm]
        
        # Idea 4: Filtros Cuantitativos 95%
        viable_quant = [c for c in combos_9_base if es_ticket_cuantil_estricto(c, recent_numbers)]
        if len(viable_quant) < 3: viable_quant = viable_9_base
        t_quant = select_optimal_wheel_triplet(viable_quant, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # Control Aleatorio
        np.random.seed(target_sorteo)
        t_azar = [tuple(sorted(np.random.choice(range(46), 6, replace=False))) for _ in range(3)]
        
        candidates = [
            ('0. Modelo Original (Pool 12, Anti-SPOF, Overlap <= 3)', t_orig, pool_12),
            ('Baseline: BIBD Pool 9 Estático (Producción)', t_base, pool_9_base),
            ('Idea 1: Clique Core (Máxima Sinergia de Pares)', t_clique, pool_9_base),
            ('Idea 2: Ensamble Dinámico por Régimen (Momentum vs Reversión)', t_dyn, pool_9_dyn),
            ('Idea 3: Especialización Multi-Modal (T1 Trad, T2 Seg, T3 SS)', t_mm, pool_9_base),
            ('Idea 4: Filtros Cuantitativos 95% (Suma 90-178, Spread 22-43)', t_quant, pool_9_base),
            ('Control: Azar Puro (3 Boletos Aleatorios)', t_azar, [])
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
            strategies[name]['boletos_3'] += sum(1 for h in hits_trad if h == 3)
            strategies[name]['boletos_4'] += sum(1 for h in hits_trad if h >= 4)
            
            if max_h_trad >= 3: strategies[name]['trad_3+'] += 1
            if max_h_trad >= 4: strategies[name]['trad_4+'] += 1
            if max_h_trad >= 5: strategies[name]['trad_5+'] += 1
            if max_any >= 4: strategies[name]['any_4+'] += 1
            if max_any >= 5: strategies[name]['any_5+'] += 1
            if max_pe >= 5: strategies[name]['pe_5+'] += 1
            if max_pe >= 6: strategies[name]['pe_6'] += 1
            
        if (step_i + 1) % 100 == 0:
            print(f" -> Progreso: {step_i + 1} de {n_eval} sorteos completados...")

    elapsed = time.time() - start_time
    print(f"\n[OK] Simulación de {n_eval} sorteos completada exitosamente en {elapsed:.1f} segundos.\n")
    
    print("="*112)
    print(f"{'Estrategia / Modelo Evaluado (500 Sorteos)':<58} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
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
        print(f"{name:<58} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}, 6: {pe6}]")
    print("="*112)
    
    print("\n" + "="*85)
    print(" ESTADÍSTICAS CRÍTICAS DE RENDIMIENTO (500 SORTEOS):")
    print("="*85)
    for name, s in strategies.items():
        if 'Azar' in name: continue
        print(f" * {name}:")
        print(f"    - Boletos individuales con 3 aciertos: {s['boletos_3']} boletos")
        print(f"    - Boletos individuales con 4+ aciertos: {s['boletos_4']} boletos")
        print(f"    - Sorteos con desparramo (Pool >= 3 pero Boletos < 3): {s['desparramados']} fechas")
        print(f"    - Premios mayores (5 aciertos en cualquier modalidad): {s['any_5+']} fechas")
    print("="*85)

if __name__ == '__main__':
    run_500_draws_benchmark()

