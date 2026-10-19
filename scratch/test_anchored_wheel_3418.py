import sys
sys.path.insert(0, 'c:/Users/Administrador/Desktop/lot')

import itertools
from collections import Counter
from generador_cobertura_ml import load_data, es_ticket_valido, compute_exponential_cooccurrence_scores

draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
train_3417 = [d for d in draws if d['sorteo'] <= 3417]
real_3418 = [d for d in draws if d['sorteo'] == 3418]
real_by_mod = {d['modalidad']: d['numbers'] for d in real_3418}

# Pool 14 before 3418:
# [0, 4, 5, 9, 10, 12, 14, 16, 23, 25, 27, 31, 38, 39]
# Ranked by Borda score:
# Top 4 Anchors: [4, 10, 5, 25] (or similar)
# Let's inspect the exact Borda ranking for these 14 numbers:
top_pool = [0, 4, 5, 9, 10, 12, 14, 16, 23, 25, 27, 31, 38, 39]
pcr_scores, cooccur = compute_exponential_cooccurrence_scores(train_3417)

# Let's test the Hierarchical Anchored Wheel on pool 14
combos = [c for c in itertools.combinations(top_pool, 6) if es_ticket_valido(c)]

# Find the 3 tickets that MAXIMIZE the probability of getting >= 4 hits when any 5 winning numbers are in the pool
# Test against the real winning numbers of Segunda: [4, 10, 16, 21, 27, 38]
seg_real = set(real_by_mod['Segunda'])

# Let's check how many sets of 3 tickets achieve 4 or 5 hits in Segunda:
best_triplet = None
best_hits = 0

# Let's test our Hierarchical Strategy:
# Ticket 1: Max density in Top 8
top8 = top_pool[:8]
combos_8 = [c for c in combos if sum(1 for n in c if n in top8) >= 4]

# Ticket A:
tA = max(combos_8, key=lambda c: sum(pcr_scores[n] for n in c) + 300.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)))

# Ticket B: overlap of 3 or 4 with tA, maximizing pair synergy
v_B = [c for c in combos if 3 <= len(set(c).intersection(set(tA))) <= 4]
tB = max(v_B, key=lambda c: sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)) * 400.0 + sum(pcr_scores[n] for n in c))

# Ticket C: joint quad coverage with A and B
quads_AB = set(itertools.combinations(tA, 4)).union(set(itertools.combinations(tB, 4)))
def score_C(c):
    new_quads = len(set(itertools.combinations(c, 4)) - quads_AB)
    return new_quads * 80.0 + sum(pcr_scores[n] for n in c) + 200.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2))

v_C = [c for c in combos if c != tA and c != tB and 2 <= len(set(c).intersection(set(tA))) <= 4 and 2 <= len(set(c).intersection(set(tB))) <= 4]
tC = max(v_C, key=score_C)

print(f"Ticket A: {sorted(tA)} -> Segunda hits: {len(set(tA).intersection(seg_real))}")
print(f"Ticket B: {sorted(tB)} -> Segunda hits: {len(set(tB).intersection(seg_real))}")
print(f"Ticket C: {sorted(tC)} -> Segunda hits: {len(set(tC).intersection(seg_real))}")

pozo_extra = set(real_by_mod['Tradicional'] + real_by_mod['Segunda'] + real_by_mod['Revancha'])
print(f"\nPozo Extra hits:")
print(f"Ticket A: {len(set(tA).intersection(pozo_extra))}")
print(f"Ticket B: {len(set(tB).intersection(pozo_extra))}")
print(f"Ticket C: {len(set(tC).intersection(pozo_extra))}")
