import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
from collections import Counter
import numpy as np
from generador_cobertura_ml import load_data, compute_antagonist_pairs

draws = load_data()
sorteos = sorted(list(set(d['sorteo'] for d in draws)))
test_sorteos = sorteos[-300:]

print("Ejecutando test rápido de Triple Overlap (Max 1 vs Max 2)...")
# Let's inspect B1, B2, B3 generation across 150 draws to compare

