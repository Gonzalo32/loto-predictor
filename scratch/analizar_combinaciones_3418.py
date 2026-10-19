import csv
import json
import numpy as np
import itertools
from collections import defaultdict, Counter

# Let's inspect Sorteo 3418 specifically
import sys
sys.path.insert(0, 'c:/Users/Administrador/Desktop/lot')

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_adaptive_pool,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_cross_modal_transitions
)

draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
train_3417 = [d for d in draws if d['sorteo'] <= 3417]
real_3418 = [d for d in draws if d['sorteo'] == 3418]
real_by_mod = {d['modalidad']: d['numbers'] for d in real_3418}

print("="*70)
print("ANÁLISIS SORTEO #3418 CON LAS PREDICCIONES DEL MODELO")
print("="*70)
print("Resultados reales:")
for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
    print(f"  {mod:12s}: {real_by_mod[mod]}")

# Top 14 pool that was generated:
top_pool = [0, 4, 5, 9, 10, 12, 14, 16, 23, 25, 27, 31, 38, 39]
seg_real = set(real_by_mod['Segunda']) # {4, 10, 16, 21, 27, 38}
hits_in_pool = sorted(list(seg_real.intersection(set(top_pool))))
print(f"\nNúmeros de Segunda en el Pool ({len(hits_in_pool)}/6): {hits_in_pool}")

# How many combinations of 6 from top_pool contain 4 or 5 hits in Segunda?
combos_pool = [c for c in itertools.combinations(top_pool, 6) if es_ticket_valido(c)]
print(f"Total combinaciones viables en el Pool: {len(combos_pool)}")

combos_with_5_hits = [c for c in combos_pool if len(set(c).intersection(seg_real)) == 5]
combos_with_4_hits = [c for c in combos_pool if len(set(c).intersection(seg_real)) == 4]
combos_with_3_hits = [c for c in combos_pool if len(set(c).intersection(seg_real)) == 3]

print(f"Combinaciones viables con 5 ACIERTOS en Segunda: {len(combos_with_5_hits)}")
print(f"Combinaciones viables con 4 ACIERTOS en Segunda: {len(combos_with_4_hits)}")
print(f"Combinaciones viables con 3 ACIERTOS en Segunda: {len(combos_with_3_hits)}")

if combos_with_5_hits:
    print(f"\nEjemplos de combinaciones con 5 aciertos:")
    for c in combos_with_5_hits[:5]:
        print(f"  {c} -> Hits: {sorted(list(set(c).intersection(seg_real)))}")

if combos_with_4_hits:
    print(f"\nEjemplos de combinaciones con 4 aciertos:")
    for c in combos_with_4_hits[:5]:
        print(f"  {c} -> Hits: {sorted(list(set(c).intersection(seg_real)))}")
