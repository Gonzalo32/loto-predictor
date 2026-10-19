import itertools
import numpy as np
from collections import defaultdict

# The 14 numbers that the ML model selected for 3417
pool_14 = [0, 1, 4, 5, 6, 9, 16, 19, 23, 27, 31, 38, 43, 44]

# Real results of 3417
sorteo_3417 = {
    'Tradicional': [1, 9, 23, 24, 32, 37],
    'Segunda': [5, 8, 13, 23, 43, 44],
    'Revancha': [4, 6, 10, 25, 30, 37],
    'SiempreSale': [1, 14, 18, 41, 43, 44]
}
pozo_extra = set(sorteo_3417['Tradicional'] + sorteo_3417['Segunda'] + sorteo_3417['Revancha'])

print("Pool 14:", pool_14)
print("Segunda real:", sorteo_3417['Segunda'])
print("Hits in pool for Segunda:", [n for n in sorteo_3417['Segunda'] if n in pool_14])

# Valid combinations within pool 14
import sys
sys.path.insert(0, 'c:/Users/Administrador/Desktop/lot')
from generador_cobertura_ml import es_ticket_valido, load_data, compute_exponential_cooccurrence_scores

draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
train_draws = [d for d in draws if d['sorteo'] <= 3416]
pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)

# All valid 6-combinations from pool_14
combos = [c for c in itertools.combinations(pool_14, 6) if es_ticket_valido(c)]
print(f"Total valid combos in pool 14: {len(combos)}")

# Check combos that have 4 hits in Segunda:
combos_4_segunda = [c for c in combos if len(set(c).intersection(set(sorteo_3417['Segunda']))) == 4]
print(f"Combos with 4 hits in Segunda in pool 14: {len(combos_4_segunda)}")
for c in combos_4_segunda[:5]:
    print("  Combo 4 hits Segunda:", c)

# Check combos that have 5 or 6 hits in Pozo Extra:
combos_5_pe = [c for c in combos if len(set(c).intersection(pozo_extra)) >= 5]
print(f"Combos with 5+ hits in Pozo Extra in pool 14: {len(combos_5_pe)}")
for c in combos_5_pe[:5]:
    print(f"  Combo {len(set(c).intersection(pozo_extra))} hits Pozo Extra:", c)
