import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter, defaultdict
import numpy as np

from generador_cobertura_ml import (
    load_data, compute_exponential_cooccurrence_scores,
    compute_delta_delay_momentum_scores, compute_attention_mlp_scores
)

draws = load_data('historico_quini_completo.csv')
train = [d for d in draws if d['sorteo'] <= 3418]

pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train)
ddm_scores = compute_delta_delay_momentum_scores(train)
attn_scores = compute_attention_mlp_scores(train)

# Frequency last 60
counts_60 = Counter()
for d in train[-60:]:
    for n in d['numbers']:
        counts_60[n] += 1

# Recency
recency_scores = np.zeros(46)
for t_idx, d in enumerate(train[-30:]):
    weight = np.exp((t_idx - 30) / 10.0)
    for n in d['numbers']:
        recency_scores[n] += weight

print("COMPARACIÓN 38 vs 39 antes del sorteo 3419:")
print(f"Freq 60:        38: {counts_60[38]} | 39: {counts_60[39]}")
print(f"Recency:        38: {recency_scores[38]:.2f} | 39: {recency_scores[39]:.2f}")
print(f"PCR (Cooccur):  38: {pcr_scores[38]:.2f} | 39: {pcr_scores[39]:.2f}")
print(f"DDM (Momentum): 38: {ddm_scores[38]:.2f} | 39: {ddm_scores[39]:.2f}")
print(f"Attention MLP:  38: {attn_scores[38]:.4f} | 39: {attn_scores[39]:.4f}")

# Look at cooccurrence with 4, 5, 44:
print("\nCo-ocurrencia histórica con el núcleo ganador [4, 5, 44]:")
for num in [38, 39]:
    co4 = cooccur_exp_mat[min(4, num), max(4, num)]
    co5 = cooccur_exp_mat[min(5, num), max(5, num)]
    co44 = cooccur_exp_mat[min(44, num), max(44, num)]
    print(f"Nro {num:02d} -> con 4: {co4:.3f}, con 5: {co5:.3f}, con 44: {co44:.3f} | Total: {(co4+co5+co44):.3f}")
