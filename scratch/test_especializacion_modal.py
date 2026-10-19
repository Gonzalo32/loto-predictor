import sys
sys.path.insert(0, 'c:/Users/Administrador/Desktop/lot')

import csv
import itertools
import numpy as np
from collections import defaultdict, Counter
from generador_cobertura_ml import load_data, es_ticket_valido

draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
train_3417 = [d for d in draws if d['sorteo'] <= 3417]
real_3418 = [d for d in draws if d['sorteo'] == 3418]
real_by_mod = {d['modalidad']: d['numbers'] for d in real_3418}

# Filtrar historiales específicos por modalidad
draws_trad = [d for d in train_3417 if d['modalidad'] == 'Tradicional']
draws_seg  = [d for d in train_3417 if d['modalidad'] == 'Segunda']
draws_ss   = [d for d in train_3417 if d['modalidad'] == 'SiempreSale']

def compute_modality_scores(mod_draws, all_draws):
    # Frecuencia en ventana
    counts_40 = Counter()
    for d in mod_draws[-40:]:
        for n in d['numbers']: counts_40[n] += 1
        
    # Recencia en esa modalidad
    recency = np.zeros(46)
    for t_idx, d in enumerate(mod_draws[-25:]):
        w = np.exp((t_idx - 25) / 8.0)
        for n in d['numbers']: recency[n] += w
        
    # Co-ocurrencia dentro de esa modalidad
    cooccur = defaultdict(lambda: defaultdict(float))
    for t_idx, d in enumerate(mod_draws[-40:]):
        w = np.exp((t_idx - 40) / 12.0)
        for u, v in itertools.combinations(d['numbers'], 2):
            cooccur[u][v] += w
            cooccur[v][u] += w

    # Delays en esa modalidad
    last_idx = {}
    for i, d in enumerate(mod_draws):
        for n in d['numbers']: last_idx[n] = i
    T = len(mod_draws)
    delays = {n: T - 1 - last_idx.get(n, 0) for n in range(46)}
    
    # Delay momentum
    delay_score = {}
    for n in range(46):
        d = delays[n]
        if 2 <= d <= 8: delay_score[n] = 2.5
        elif d == 1: delay_score[n] = 1.0 # repetición consecutiva
        elif d > 20: delay_score[n] = -2.0
        else: delay_score[n] = 0.5

    borda = {}
    for n in range(46):
        borda[n] = counts_40[n] * 1.6 + recency[n] * 2.2 + delay_score[n] * 1.5
    return borda, cooccur

borda_trad, cooccur_trad = compute_modality_scores(draws_trad, train_3417)
borda_seg, cooccur_seg   = compute_modality_scores(draws_seg, train_3417)

rank_trad = sorted(range(46), key=lambda x: borda_trad[x], reverse=True)
rank_seg  = sorted(range(46), key=lambda x: borda_seg[x], reverse=True)

print("="*70)
print("TOP 12 ESPECIALIZADO PARA TRADICIONAL:")
print(f"Top 12 Trad: {rank_trad[:12]}")
hits_trad = [n for n in real_by_mod['Tradicional'] if n in rank_trad[:12]]
print(f"Aciertos en Top 12 Tradicional ({len(hits_trad)}/6): {hits_trad} (Reales: {real_by_mod['Tradicional']})")

print("\nTOP 12 ESPECIALIZADO PARA SEGUNDA:")
print(f"Top 12 Segunda: {rank_seg[:12]}")
hits_seg = [n for n in real_by_mod['Segunda'] if n in rank_seg[:12]]
print(f"Aciertos en Top 12 Segunda ({len(hits_seg)}/6): {hits_seg} (Reales: {real_by_mod['Segunda']})")
