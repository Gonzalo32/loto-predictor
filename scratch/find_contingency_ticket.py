import os, sys, itertools, math
from collections import Counter
import numpy as np

WORKSPACE_DIR = r'c:\Users\Administrador\Desktop\lot'
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import load_data, compute_exponential_cooccurrence_scores, compute_antagonist_pairs, es_ticket_valido

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
trad = [d for d in draws if d['modalidad'] == 'Tradicional']

b1 = (0, 1, 6, 23, 25, 43)
b2 = (0, 1, 23, 39, 40, 43)
b3 = (0, 6, 9, 10, 23, 31)

p123 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2)) | set(itertools.combinations(b3, 2))

t_pool = [0, 1, 4, 5, 6, 9, 10, 14, 21, 23, 25, 31, 32, 39, 40, 43]
cands_reserva = []
for c in itertools.combinations(t_pool, 6):
    if c in (b1, b2, b3): continue
    if not es_ticket_valido(c, min_distinct_decades=3): continue
    new_pairs = len(set(itertools.combinations(c, 2)) - p123)
    cands_reserva.append((c, new_pairs))

cands_reserva.sort(key=lambda x: x[1], reverse=True)
best_reserve, n_pairs = cands_reserva[0]
print(f"Boleto de Reserva / Plan B Óptimo: {[f'{n:02d}' for n in sorted(best_reserve)]} (Aporta {n_pairs} pares nuevos no cubiertos)")

