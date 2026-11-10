import os, sys, itertools, csv
from collections import defaultdict, Counter
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_adaptive_pool,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_attention_mlp_scores
)
from cobertura_combinatoria import select_optimal_wheel_triplet, select_balanced_bibd_triplet

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]

# -------------------------------------------------------------
# IDEA 1: Trío Núcleo por Máxima Sinergia de Coocurrencia (Clique)
# En vez de pool[:3], buscar el trío dentro del pool 9 que maximice
# la suma de coocurrencia entre sus 3 pares: cooccur(A,B) + cooccur(A,C) + cooccur(B,C)
# -------------------------------------------------------------
def select_bibd_clique_core(pool_9, borda, cooccur_matrix, boletos_viables=None, w_cooccur=350.0):
    # Evaluar los C(9, 3) = 84 tríos posibles
    all_trios = list(itertools.combinations(pool_9, 3))
    
    def score_trio(trio):
        u, v, w = trio
        node_s = borda.get(u, 0) + borda.get(v, 0) + borda.get(w, 0)
        edge_s = cooccur_matrix[u, v] + cooccur_matrix[u, w] + cooccur_matrix[v, w]
        return node_s * 0.4 + edge_s * 400.0
        
    best_trio = max(all_trios, key=score_trio)
    core = list(best_trio)
    rest = [n for n in pool_9 if n not in core]
    
    core_pairs = [
        (core[0], core[1]),
        (core[0], core[2]),
        (core[1], core[2])
    ]
    
    pairs_of_rest = list(itertools.combinations(rest, 2))
    partitions = []
    for p1 in pairs_of_rest:
        rem1 = [n for n in rest if n not in p1]
        for p2 in itertools.combinations(rem1, 2):
            p3 = tuple(sorted([n for n in rem1 if n not in p2]))
            part = tuple(sorted([p1, p2, p3]))
            if part not in partitions:
                partitions.append(part)
                
    best_triplet = None
    best_score = -float('inf')
    viable_set = set(tuple(sorted(c)) for c in boletos_viables) if boletos_viables else None
    
    for part in partitions:
        for perm in itertools.permutations(part):
            sub_rest1 = [n for n in rest if n not in perm[0]]
            t1 = tuple(sorted(list(core_pairs[0]) + sub_rest1))
            if viable_set is not None and t1 not in viable_set: continue
            
            sub_rest2 = [n for n in rest if n not in perm[1]]
            t2 = tuple(sorted(list(core_pairs[1]) + sub_rest2))
            if viable_set is not None and t2 not in viable_set: continue
            
            sub_rest3 = [n for n in rest if n not in perm[2]]
            t3 = tuple(sorted(list(core_pairs[2]) + sub_rest3))
            if viable_set is not None and t3 not in viable_set: continue
            
            total_score = 0.0
            for t in [t1, t2, t3]:
                total_score += sum(borda.get(n, 0) for n in t) + w_cooccur * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(t, 2))
                
            if total_score > best_score:
                best_score = total_score
                best_triplet = [t1, t2, t3]
                
    if best_triplet is None:
        # Fallback a la selección estándar
        return select_optimal_wheel_triplet(boletos_viables or [], pool_9, borda, cooccur_matrix)
    return best_triplet

# -------------------------------------------------------------
# Test de comparación en los 50 sorteos
# -------------------------------------------------------------
results = {
    'Modelo Actual (BIBD Pool 9 Top 3 Individual)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_hits': []},
    'Idea 1: BIBD con Trío Núcleo Clique (Máxima Sinergia de Pares)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_hits': []},
}

for target_draw in eval_draws:
    target_sorteo = target_draw['sorteo']
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
        w = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency_scores[n] += w
    pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
    ddm_scores = compute_delta_delay_momentum_scores(train_draws)
    attn_scores = compute_attention_mlp_scores(train_draws)
    borda = defaultdict(float)
    models_to_rank = [
        (counts_60, 1.5), (recency_scores, 2.0), (pcr_scores, 1.4), (ddm_scores, 1.8),
        ({n: attn_scores[n] for n in range(46)}, 1.5)
    ]
    for m_scores, w in models_to_rank:
        ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
        for rank, n in enumerate(ranked):
            borda[n] += w * (45 - rank)
    ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)
    
    pool_9 = sorted(ranking[:9])
    recent_numbers = set()
    for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
    
    combos_9 = list(itertools.combinations(pool_9, 6))
    viable_9 = [c for c in combos_9 if es_ticket_valido(c, recent_numbers)] or combos_9
    
    # 1. Actual
    t_actual = select_optimal_wheel_triplet(viable_9, pool_9, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    
    # 2. Idea 1 Clique Core
    t_clique = select_bibd_clique_core(pool_9, borda, cooccur_exp_mat, viable_9, w_cooccur=350.0)
    
    for name, t_list in [('Modelo Actual (BIBD Pool 9 Top 3 Individual)', t_actual),
                         ('Idea 1: BIBD con Trío Núcleo Clique (Máxima Sinergia de Pares)', t_clique)]:
        h_trad = [len(set(t) & target_trad) for t in t_list]
        max_h = max(h_trad)
        max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
        max_pe = max(len(set(t) & pozo_extra) for t in t_list)
        
        results[name]['max_hits'].append(max_h)
        if max_h >= 3: results[name]['trad_3+'] += 1
        if max_h >= 4: results[name]['trad_4+'] += 1
        if max_any >= 4: results[name]['any_4+'] += 1
        if max_pe >= 5: results[name]['pe_5+'] += 1

print("="*85)
print(" RESULTADOS COMPARATIVOS: BIBD BASE VS BIBD CLIQUE CORE (50 SORTEOS)")
print("="*85)
for name, data in results.items():
    print(f"{name}:")
    print(f"  - Promedio acierto máximo Tradicional: {np.mean(data['max_hits']):.2f}")
    print(f"  - Sorteos con 3+ aciertos (Tradicional): {data['trad_3+']} de 50 ({(data['trad_3+']/50)*100:.1f}%)")
    print(f"  - Sorteos con 4+ aciertos (Cualquier mod.): {data['any_4+']} de 50 ({(data['any_4+']/50)*100:.1f}%)")
    print(f"  - Sorteos con 5+ aciertos en Pozo Extra: {data['pe_5+']} de 50 ({(data['pe_5+']/50)*100:.1f}%)")
print("="*85)

