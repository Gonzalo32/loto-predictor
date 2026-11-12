import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
from collections import Counter
import numpy as np
from generador_cobertura_ml import load_data

# Let's inspect how the top ranking changes when bonus_res is 6.0 vs 18.0 on sorteo 3425
draws = load_data()
print(f"Total sub-draws: {len(draws)}")
last_s = max(d['sorteo'] for d in draws)
print(f"Ultimo sorteo: {last_s}")

