import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
from collections import Counter
import numpy as np
from generador_cobertura_ml import load_data, compute_antagonist_pairs

# Load draws
draws = load_data()
sorteos = sorted(list(set(d['sorteo'] for d in draws)))

# We will run a 600-draw backtest comparing:
# Strategy 1 (Current Champion): 2-Fold BIBD (overlap in [3, 4], some balls appear in all 3 tickets)
# Strategy 2 (Max 2 Appearances): No ball is allowed to appear in all 3 tickets (Max ball count in coupon <= 2)
# Strategy 3 (Max 1 Common Anchor): At most 1 anchor ball allowed in all 3 tickets; all other balls in at most 2 tickets

print("Iniciando backtest de solapamiento y riesgo de concentración en 300 sorteos...")
# Let's run a test on 300 draws first to get quick, clean results
test_sorteos = sorteos[-300:]

# Let's inspect how B1, B2, B3 are formed in generador_cobertura_ml

