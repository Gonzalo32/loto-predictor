import os, sys, itertools
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.getcwd())
from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_exponential_cooccurrence_scores,
    compute_delta_delay_momentum_scores, compute_attention_mlp_scores
)

draws = load_data('historico_quini_completo.csv')
train = [d for d in draws if d['sorteo'] <= 3418]
s3419 = [d for d in draws if d['sorteo'] == 3419]
trad_real = [d['numbers'] for d in s3419 if d['modalidad'] == 'Tradicional'][0]

top_pool = [4, 5, 6, 10, 14, 15, 23, 28, 38, 39, 41, 44]
pcr_scores, cooccur = compute_exponential_cooccurrence_scores(train)

# Normalize scores as proxy probabilities
max_pcr = max(pcr_scores[n] for n in top_pool) or 1.0
prob_proxy = {n: (pcr_scores[n] / max_pcr) for n in top_pool}

# Compute quartet weight for each possible quartet of pool
quartet_weights = {}
for q in itertools.combinations(top_pool, 4):
    edge_sum = sum(cooccur[u, v] for u, v in itertools.combinations(q, 2))
    node_prod = np.prod([prob_proxy[n] for n in q])
    quartet_weights[q] = edge_sum * 10.0 + node_prod

combos = [c for c in itertools.combinations(top_pool, 6) if es_ticket_valido(c)]

# For each combo, precompute its 15 quartets and its individual combo score
combo_quartets = [set(itertools.combinations(c, 4)) for c in combos]
combo_scores = [sum(pcr_scores[n] for n in c) + 300.0 * sum(cooccur[u, v] for u, v in itertools.combinations(c, 2)) for c in combos]

# Score each triplet by:
# Sum of quartet weights covered + combo quality score
best_trip = None
best_score = -1.0

# Pre-filter top 150 combos
indexed = sorted(range(len(combos)), key=lambda i: combo_scores[i], reverse=True)[:150]

pool_set = set(top_pool)

for idx_1, i in enumerate(indexed[:25]):
    q1 = combo_quartets[i]
    c1 = combos[i]
    c1_set = set(c1)
    
    for idx_2, j in enumerate(indexed[idx_1 + 1:75]):
        c2 = combos[j]
        c2_set = set(c2)
        if len(c1_set.intersection(c2_set)) > 3:
            continue
            
        covered_12 = c1_set.union(c2_set)
        uncovered = pool_set - covered_12
        if len(uncovered) > 6:
            continue
            
        q2 = combo_quartets[j]
        q12 = q1.union(q2)
        
        for k in indexed[idx_2 + 1:]:
            c3 = combos[k]
            c3_set = set(c3)
            
            # Full pool coverage constraint
            if not uncovered.issubset(c3_set):
                continue
                
            if len(c1_set.intersection(c3_set)) > 3 or len(c2_set.intersection(c3_set)) > 3:
                continue
                
            # Anti SPOF
            if len(c1_set.intersection(c2_set).intersection(c3_set)) > 0:
                continue
                
            q3 = combo_quartets[k]
            all_quads = q12.union(q3)
            
            # Total weighted coverage of quartets
            w_cov = sum(quartet_weights[q] for q in all_quads)
            score = w_cov + (combo_scores[i] + combo_scores[j] + combo_scores[k]) * 0.1
            
            if score > best_score:
                best_score = score
                best_trip = [c1, c2, c3]

print(f"Mejor terna encontrada (Score: {best_score:.2f}):")
for idx, t in enumerate(best_trip):
    hits = set(t).intersection(set(trad_real))
    print(f"Ticket {idx+1}: {sorted(t)} -> {len(hits)} aciertos en Tradicional: {sorted(list(hits))}")
