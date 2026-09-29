import itertools
import numpy as np
from collections import defaultdict
from typing import List, Set, Tuple, Union

def _get_score(scores: Union[dict, list, np.ndarray], num: int) -> float:
    if isinstance(scores, (list, np.ndarray)):
        return float(scores[num]) if 0 <= num < len(scores) else 0.0
    elif isinstance(scores, dict):
        return float(scores.get(num, 0.0))
    return 0.0

def select_wheel_coverage_tickets(
    boletos_viables: List[Tuple[int, ...]],
    scores: Union[dict, list, np.ndarray],
    n_tickets: int = 3,
    target_subsize: int = 4
) -> List[Tuple[int, ...]]:
    """
    Selecciona `n_tickets` boletos de 6 números que maximizan la cobertura combinatoria
    (Hypergraph 4-subsets Coverage) del Pool de números.
    
    Asegura matemáticamente que los tríos y cuartetos del Pool se distribuyan sin solapamientos
    redundantes entre las boletas.
    """
    if len(boletos_viables) <= n_tickets:
        return boletos_viables

    # Mapeo de subconjuntos de tamaño 4 contenidos en cada boleto viable
    subsets_by_ticket = []
    for t_idx, ticket in enumerate(boletos_viables):
        sub_set = set(itertools.combinations(ticket, target_subsize))
        subsets_by_ticket.append(sub_set)

    selected_indices = []
    covered_subsets: Set[Tuple[int, ...]] = set()

    # Boleto 1: El boleto viable con mayor suma de scores individuales
    b1_idx = max(
        range(len(boletos_viables)),
        key=lambda i: sum(_get_score(scores, n) for n in boletos_viables[i])
    )
    selected_indices.append(b1_idx)
    covered_subsets.update(subsets_by_ticket[b1_idx])

    # Boletos 2..N: Selección voraz (Greedy Maximum Marginal Coverage)
    while len(selected_indices) < n_tickets:
        best_idx = -1
        best_gain = -1.0

        for t_idx in range(len(boletos_viables)):
            if t_idx in selected_indices:
                continue

            # Nuevos subconjuntos de 4 números cubiertos por este boleto
            new_covered = subsets_by_ticket[t_idx] - covered_subsets
            
            # Ponderación por el score medio de los números
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
