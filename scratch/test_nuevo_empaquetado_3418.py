import sys
sys.path.insert(0, 'c:/Users/Administrador/Desktop/lot')

import itertools
import numpy as np
from collections import defaultdict, Counter
from generador_cobertura_ml import (
    load_data, es_ticket_valido,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_cross_modal_transitions
)

draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
train_3417 = [d for d in draws if d['sorteo'] <= 3417]
real_3418 = [d for d in draws if d['sorteo'] == 3418]
real_by_mod = {d['modalidad']: d['numbers'] for d in real_3418}

# Pool from 3418
top_pool = [0, 4, 5, 9, 10, 12, 14, 16, 23, 25, 27, 31, 38, 39]
pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_3417)
ddm_scores = compute_delta_delay_momentum_scores(train_3417)

# Borda weights
counts_60 = Counter()
for d in train_3417[-60:]:
    for n in d['numbers']: counts_60[n] += 1
recency = np.zeros(46)
for t_idx, d in enumerate(train_3417[-30:]):
    w = np.exp((t_idx - 30) / 10.0)
    for n in d['numbers']: recency[n] += w

borda = {}
for n in range(46):
    borda[n] = counts_60[n] * 1.5 + recency[n] * 2.0 + pcr_scores[n] * 1.4 + ddm_scores[n] * 1.0

combos_viables = [c for c in itertools.combinations(top_pool, 6) if es_ticket_valido(c)]

print("="*70)
print("TEST DE NUEVO EMPAQUETADO CONCENTRADO (PREMIO-ORIENTED) EN SORTEO 3418")
print("="*70)
print(f"Resultados reales Segunda: {real_by_mod['Segunda']}")
print(f"Resultados reales Tradicional: {real_by_mod['Tradicional']}")

# NUEVO EMPAQUETADOR:
# En vez de forzar dispersión (overlap <= 3 y borda_C * 0.5):
# 1. Ticket A (Densidad Core): Maximiza score de Borda + co-ocurrencia
scored_combos = []
for c in combos_viables:
    score = sum(borda[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
    scored_combos.append((score, c))
scored_combos.sort(key=lambda x: x[0], reverse=True)

tA_nuevo = scored_combos[0][1]

# 2. Ticket B (Sinergia Segunda / Concentración de Pares):
# Permite solapamiento de 3 a 4 números con A para NO romper los cuartetos de alta probabilidad,
# pero rota 2 números para capturar los candidatos complementarios del Core
seg_counts = Counter()
for d in train_3417[-40:]:
    if d['modalidad'] == 'Segunda':
        for n in d['numbers']: seg_counts[n] += 1

v_B = [c for c in combos_viables if 2 <= len(set(c).intersection(set(tA_nuevo))) <= 4]
scored_B = []
for c in v_B:
    s = sum(borda[n] + 15.0 * seg_counts[n] for n in c) + 400.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
    scored_B.append((s, c))
scored_B.sort(key=lambda x: x[0], reverse=True)
tB_nuevo = scored_B[0][1]

# 3. Ticket C (Rueda de Cobertura de Cuartetos sin penalizar números buenos):
# Maximiza la cobertura de cuartetos no cubiertos por A y B, pero usando números con alto score
quads_AB = set(itertools.combinations(tA_nuevo, 4)).union(set(itertools.combinations(tB_nuevo, 4)))
scored_C = []
for c in combos_viables:
    if c == tA_nuevo or c == tB_nuevo: continue
    quads_c = set(itertools.combinations(c, 4))
    new_quads = len(quads_c - quads_AB)
    # No pulverizar el score de números de A y B: sólo bonificar cuartetos frescos
    s = sum(borda[n] for n in c) + new_quads * 60.0 + 300.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
    scored_C.append((s, c))
scored_C.sort(key=lambda x: x[0], reverse=True)
tC_nuevo = scored_C[0][1]

print("\n--- TICKETS GENERADOS CON NUEVO EMPAQUETADO ---")
tickets_nuevos = [('Ticket A (Core)', tA_nuevo), ('Ticket B (Segunda/Cluster)', tB_nuevo), ('Ticket C (Quad-Cover)', tC_nuevo)]

for name, t in tickets_nuevos:
    t_set = set(t)
    h_tr = len(t_set.intersection(set(real_by_mod['Tradicional'])))
    h_sg = len(t_set.intersection(set(real_by_mod['Segunda'])))
    h_rv = len(t_set.intersection(set(real_by_mod['Revancha'])))
    h_ss = len(t_set.intersection(set(real_by_mod['SiempreSale'])))
    pe_set = set(real_by_mod['Tradicional'] + real_by_mod['Segunda'] + real_by_mod['Revancha'])
    h_pe = len(t_set.intersection(pe_set))
    print(f"{name:25s}: {sorted(list(t))} | Trad:{h_tr} | Seg:{h_sg} | Rev:{h_rv} | SS:{h_ss} | PozoExtra:{h_pe}")
