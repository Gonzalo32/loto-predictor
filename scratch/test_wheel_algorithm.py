import itertools
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Set, Dict, Optional

def select_optimal_wheel_triplet(
    boletos_viables: List[Tuple[int, ...]],
    pool: List[int],
    borda_scores: Dict[int, float],
    cooccur_matrix: np.ndarray,
    max_overlap: int = 3,
    w_cooccur: float = 300.0,
    top_candidates_limit: int = 250
) -> List[Tuple[int, ...]]:
    """
    Selecciona la terna óptima de 3 boletos con:
    1. Cobertura Total del Pool (todos los números del pool aparecen al menos 1 vez).
    2. Solapamiento estrictamente acotado (intersección entre cualquier par <= max_overlap).
    3. Cero repetición triple (Anti-SPOF: ningún número se juega en los 3 boletos).
    4. Máxima sinergia conjunta de Borda, coocurrencias y cobertura de 4-subsets (cuartetos).
    """
    pool_set = set(pool)
    pool_size = len(pool)
    
    # 1. Puntuar cada combinación individualmente
    scored_combos = []
    for c in boletos_viables:
        s_borda = sum(borda_scores.get(n, 0.0) for n in c)
        s_co = sum(cooccur_matrix[u, v] for u, v in itertools.combinations(c, 2))
        total_s = s_borda + w_cooccur * s_co
        scored_combos.append((total_s, c))
        
    scored_combos.sort(key=lambda x: x[0], reverse=True)
    
    # Reducir el espacio de búsqueda a los mejores candidatos para eficiencia
    top_combos = scored_combos[:top_candidates_limit]
    
    # Precomputar sets de números y cuartetos
    combo_sets = [set(c) for _, c in top_combos]
    combo_scores = [s for s, _ in top_combos]
    combo_tuples = [c for _, c in top_combos]
    
    best_triplet = None
    best_objective = -float('inf')
    
    # Búsqueda estructurada de la terna global
    # T1 siempre arranca de los boletos de más alta puntuación (núcleo)
    for i in range(min(30, len(top_combos))):
        s1 = combo_scores[i]
        c1_set = combo_sets[i]
        c1 = combo_tuples[i]
        
        for j in range(i + 1, min(120, len(top_combos))):
            c2_set = combo_sets[j]
            c2 = combo_tuples[j]
            ov12 = len(c1_set.intersection(c2_set))
            if ov12 > max_overlap:
                continue
                
            covered_12 = c1_set.union(c2_set)
            uncovered = pool_set - covered_12
            # Si el tercer boleto (de 6 números) no alcanza a cubrir los huérfanos, descartar
            if len(uncovered) > 6:
                continue
                
            s2 = combo_scores[j]
            
            for k in range(j + 1, len(top_combos)):
                c3_set = combo_sets[k]
                
                # Regla de Cobertura Total: c3 debe cubrir todos los que faltan
                if not uncovered.issubset(c3_set):
                    continue
                    
                ov13 = len(c1_set.intersection(c3_set))
                if ov13 > max_overlap:
                    continue
                    
                ov23 = len(c2_set.intersection(c3_set))
                if ov23 > max_overlap:
                    continue
                    
                # Regla Anti-SPOF: ningún número en los 3 boletos a la vez
                triple_overlap = c1_set.intersection(c2_set).intersection(c3_set)
                if len(triple_overlap) > 0:
                    continue
                    
                s3 = combo_scores[k]
                
                # Función objetivo combinada:
                # 1. Suma de scores de los 3 boletos
                # 2. Diversidad de cuartetos (Hypergraph coverage: 15*3 = 45 cuando overlap <= 3)
                # 3. Penalización por desbalance de frecuencias
                freqs = Counter(list(c1) + list(c2) + list(combo_tuples[k]))
                # Deseamos que los números de mayor Borda aparezcan 2 veces, y los menores 1 vez
                borda_weight_boost = sum(borda_scores.get(n, 0.0) * freqs[n] for n in pool_set)
                
                objective = s1 + s2 + s3 + borda_weight_boost
                
                if objective > best_objective:
                    best_objective = objective
                    best_triplet = [c1, combo_tuples[j], combo_tuples[k]]
                    
    # Fallback si no hay terna que cumpla 100% de cobertura exacta
    if best_triplet is None:
        print("[WARN] No se encontró terna con cobertura 100% estricta en el subset; aplicando Greedy Maximum Marginal Coverage...")
        t1 = top_combos[0][1]
        t1_set = set(t1)
        
        # t2 con overlap <= max_overlap
        cand_t2 = [c for _, c in top_combos[1:] if len(set(c).intersection(t1_set)) <= max_overlap]
        t2 = cand_t2[0] if cand_t2 else top_combos[1][1]
        t2_set = set(t2)
        
        covered = t1_set.union(t2_set)
        uncovered = pool_set - covered
        
        cand_t3 = []
        for s, c in top_combos:
            if c == t1 or c == t2:
                continue
            c_set = set(c)
            if len(c_set.intersection(t1_set)) <= max_overlap and len(c_set.intersection(t2_set)) <= max_overlap:
                new_cov = len(c_set.intersection(uncovered))
                cand_t3.append((new_cov * 200.0 + s, c))
                
        if cand_t3:
            cand_t3.sort(key=lambda x: x[0], reverse=True)
            t3 = cand_t3[0][1]
        else:
            t3 = top_combos[2][1]
            
        best_triplet = [t1, t2, t3]
        
    return best_triplet
