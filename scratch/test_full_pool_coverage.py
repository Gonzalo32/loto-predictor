import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
from collections import Counter, defaultdict
import numpy as np

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_exponential_cooccurrence_scores,
    compute_delta_delay_momentum_scores, compute_attention_mlp_scores
)

draws = load_data('historico_quini_completo.csv')
train = [d for d in draws if d['sorteo'] <= 3418]
s3419_real = [d for d in draws if d['sorteo'] == 3419]
real_by_mod = {d['modalidad']: d['numbers'] for d in s3419_real}
trad_real = real_by_mod['Tradicional']

# Original Pool 12:
top_pool = [4, 5, 6, 10, 14, 15, 23, 28, 38, 39, 41, 44]
pcr_scores, cooccur = compute_exponential_cooccurrence_scores(train)

# Combos viables
combos = [c for c in itertools.combinations(top_pool, 6) if es_ticket_valido(c)]

def score_combo(c):
    return sum(pcr_scores[n] for n in c) + 320.0 * sum(cooccur[u, v] for u, v in itertools.combinations(c, 2))

# Let's test Strategy:
# 1. Ticket A: Highest scoring valid combo
# 2. Overlap constraint: Ticket B and Ticket C must have overlap <= 3 with each other (to prevent duplicate quartets)
# 3. Full Coverage Constraint: All 12 numbers of top_pool MUST appear in at least one ticket!

scored_combos = sorted([(score_combo(c), c) for c in combos], key=lambda x: x[0], reverse=True)

print(f"Total combinaciones viables en pool 12: {len(combos)}")

# Search for the best 3-ticket sets that achieve FULL POOL COVERAGE (all 12 numbers covered)
# and maximum score while keeping pairwise overlap <= 3
best_sets = []
tA = scored_combos[0][1] # Best ticket

for _, tB in scored_combos[1:]:
    ov_AB = len(set(tA).intersection(set(tB)))
    if ov_AB > 3:
        continue
    covered_AB = set(tA).union(set(tB))
    uncovered = set(top_pool) - covered_AB
    if len(uncovered) > 6:
        continue # Ticket C has only 6 numbers, can't cover more than 6 uncovered
        
    for _, tC in scored_combos[1:]:
        if tC == tA or tC == tB:
            continue
        ov_AC = len(set(tA).intersection(set(tC)))
        ov_BC = len(set(tB).intersection(set(tC)))
        if ov_AC > 3 or ov_BC > 3:
            continue
        # Does tC cover all remaining uncovered numbers?
        if not uncovered.issubset(set(tC)):
            continue
            
        total_score = score_combo(tA) + score_combo(tB) + score_combo(tC)
        best_sets.append((total_score, [tA, tB, tC]))

best_sets.sort(key=lambda x: x[0], reverse=True)

print(f"\nSe encontraron {len(best_sets)} ternas con COBERTURA 100% DEL POOL (las 12 cifras cubiertas) y solapamiento <= 3:")

print(f"\nResultados Reales Tradicional: {trad_real}")
for i, (score, triplet) in enumerate(best_sets[:10]):
    hits = [len(set(t).intersection(set(trad_real))) for t in triplet]
    max_hit = max(hits)
    print(f"\n--- Opción #{i+1} (Score: {score:.1f}) | Aciertos Tradicional: {hits} -> MAX: {max_hit} ---")
    for idx, t in enumerate(triplet):
        m = sorted(list(set(t).intersection(set(trad_real))))
        print(f"  T{idx+1}: {sorted(t)} -> {len(m)} aciertos: {m}")
