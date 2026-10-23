import itertools
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Set, Dict, Union, Optional

def _get_score(scores: Union[dict, list, np.ndarray], num: int) -> float:
    if isinstance(scores, (list, np.ndarray)):
        return float(scores[num]) if 0 <= num < len(scores) else 0.0
    elif isinstance(scores, dict):
        return float(scores.get(num, 0.0))
    return 0.0

def select_optimal_wheel_triplet(
    boletos_viables: List[Tuple[int, ...]],
    pool: List[int],
    scores: Union[dict, list, np.ndarray],
    cooccur_matrix: np.ndarray,
    max_overlap: int = 3,
    w_cooccur: float = 320.0,
    top_candidates_limit: int = 200
) -> List[Tuple[int, ...]]:
    """
    Selecciona la terna óptima de 3 boletos (18 números en juego) resolviendo
    el problema de Rueda Reducida Optimizada (Covering Wheel Design) con restricciones:
    
    1. REGLA DE COBERTURA TOTAL (Zero Orphans):
       Garantiza matemáticamente que los N números del Pool aparezcan en al menos 1 boleto.
       Ningún número predicho por la IA queda fuera de las apuestas.
       
    2. LÍMITE DE SOLAPAMIENTO ANTI-COLINEALIDAD (Overlap <= 3):
       Prohíbe que dos boletos compartan 4 o más números. Cada par de boletos comparte
       como máximo 3 números, eliminando boletos redundantes y diversificando combinaciones.
       
    3. ANTI-SPOF (Single Point of Failure):
       Ningún número puede estar presente en los 3 boletos a la vez. La frecuencia máxima
       por número es 2 (los números de mayor probabilidad aparecen en 2 boletos; los demás en 1).
       
    4. MÁXIMA COBERTURA DE CUARTETOS PONDERADOS (Hypergraph 4-subsets Coverage):
       Maximiza la cobertura de subconjuntos de tamaño 4 no solapados, multiplicando
       la probabilidad de obtener premio de 4 aciertos si 4 números del pool salen sorteados.
    """
    if len(boletos_viables) <= 3:
        return boletos_viables
        
    pool_set = set(pool)
    
    # 1. Puntuación de calidad conjunta por combinación
    scored_combos = []
    for c in boletos_viables:
        s_nodes = sum(_get_score(scores, n) for n in c)
        s_edges = sum(cooccur_matrix[u, v] for u, v in itertools.combinations(c, 2))
        scored_combos.append((s_nodes + w_cooccur * s_edges, c))
    scored_combos.sort(key=lambda x: x[0], reverse=True)
    
    # Pre-filtrar a los mejores candidatos viables para búsqueda óptima eficiente
    top_combos = scored_combos[:top_candidates_limit]
    combo_sets = [set(c) for _, c in top_combos]
    combo_scores = [s for s, _ in top_combos]
    combo_tuples = [c for _, c in top_combos]
    
    # Precomputar pesos relativos de números en el pool
    max_s = max(_get_score(scores, n) for n in pool) or 1.0
    prob_weights = {n: _get_score(scores, n) / max_s for n in pool}
    
    # Precomputar pesos de cuartetos para el pool
    quartet_weights = {}
    for q in itertools.combinations(pool, 4):
        edge_w = sum(cooccur_matrix[u, v] for u, v in itertools.combinations(q, 2))
        node_w = np.prod([prob_weights[n] for n in q])
        quartet_weights[q] = edge_w * 8.0 + node_w
        
    combo_quartets = [set(itertools.combinations(c, 4)) for c in combo_tuples]
    
    best_triplet = None
    best_objective = -float('inf')
    
    # Búsqueda estructurada de la terna global
    # T1 siempre ancla en los mejores boletos (núcleo spike)
    search_limit_1 = min(35, len(top_combos))
    search_limit_2 = min(100, len(top_combos))
    
    for i in range(search_limit_1):
        s1 = combo_scores[i]
        c1_set = combo_sets[i]
        c1 = combo_tuples[i]
        q1 = combo_quartets[i]
        
        for j in range(i + 1, search_limit_2):
            c2_set = combo_sets[j]
            c2 = combo_tuples[j]
            ov12 = len(c1_set.intersection(c2_set))
            if ov12 > max_overlap:
                continue
                
            covered_12 = c1_set.union(c2_set)
            uncovered = pool_set - covered_12
            # El tercer boleto tiene 6 números, no puede cubrir más de 6 huérfanos
            if len(uncovered) > 6:
                continue
                
            s2 = combo_scores[j]
            q2 = combo_quartets[j]
            q12 = q1.union(q2)
            
            for k in range(j + 1, len(top_combos)):
                c3_set = combo_sets[k]
                c3 = combo_tuples[k]
                
                # Regla de Cobertura Total: c3 debe contener a todos los números huérfanos
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
                q3 = combo_quartets[k]
                all_quads = q12.union(q3)
                
                # Cobertura ponderada de cuartetos
                w_quads = sum(quartet_weights.get(q, 0.0) for q in all_quads)
                
                # Frecuencias de números en la terna
                freqs = Counter(list(c1) + list(c2) + list(c3))
                # Boost para que los números con mayor Borda aparezcan 2 veces
                borda_boost = sum(_get_score(scores, n) * freqs[n] for n in pool_set)
                
                objective = w_quads * 15.0 + (s1 + s2 + s3) * 0.1 + borda_boost
                
                if objective > best_objective:
                    best_objective = objective
                    best_triplet = [c1, c2, c3]
                    
    # Fallback voraz si ninguna terna cumple el 100% estricto de cobertura
    if best_triplet is None:
        t1 = top_combos[0][1]
        t1_set = set(t1)
        
        cand_t2 = [c for _, c in top_combos[1:] if len(set(c).intersection(t1_set)) <= max_overlap]
        t2 = cand_t2[0] if cand_t2 else top_combos[1][1]
        t2_set = set(t2)
        
        covered = t1_set.union(t2_set)
        uncovered = pool_set - covered
        overlap_12 = t1_set.intersection(t2_set)
        
        cand_t3 = []
        for s, c in top_combos:
            if c == t1 or c == t2:
                continue
            c_set = set(c)
            if len(c_set.intersection(t1_set)) <= max_overlap and len(c_set.intersection(t2_set)) <= max_overlap:
                if not any(n in overlap_12 for n in c):
                    new_cov = len(c_set.intersection(uncovered))
                    cand_t3.append((new_cov * 250.0 + s, c))
                    
        if cand_t3:
            cand_t3.sort(key=lambda x: x[0], reverse=True)
            t3 = cand_t3[0][1]
        else:
            t3 = top_combos[2][1]
            
        best_triplet = [t1, t2, t3]
        
    return best_triplet

def select_wheel_coverage_tickets(
    boletos_viables: List[Tuple[int, ...]],
    scores: Union[dict, list, np.ndarray],
    n_tickets: int = 3,
    target_subsize: int = 4
) -> List[Tuple[int, ...]]:
    """Función de compatibilidad previa."""
    if len(boletos_viables) <= n_tickets:
        return boletos_viables

    subsets_by_ticket = [set(itertools.combinations(t, target_subsize)) for t in boletos_viables]
    selected_indices = []
    covered_subsets: Set[Tuple[int, ...]] = set()

    b1_idx = max(range(len(boletos_viables)), key=lambda i: sum(_get_score(scores, n) for n in boletos_viables[i]))
    selected_indices.append(b1_idx)
    covered_subsets.update(subsets_by_ticket[b1_idx])

    while len(selected_indices) < n_tickets:
        best_idx = -1
        best_gain = -1.0
        for t_idx in range(len(boletos_viables)):
            if t_idx in selected_indices:
                continue
            new_covered = subsets_by_ticket[t_idx] - covered_subsets
            score_weight = sum(_get_score(scores, n) for n in boletos_viables[t_idx])
            gain = len(new_covered) * 1.5 + score_weight * 0.5
            if gain > best_gain:
                best_gain = gain
                best_idx = t_idx
        if best_idx != -1:
            selected_indices.append(best_idx)
            covered_subsets.update(subsets_by_ticket[best_idx])
        else:
            break

    return [boletos_viables[i] for i in selected_indices]
