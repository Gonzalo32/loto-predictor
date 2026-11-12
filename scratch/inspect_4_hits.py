import os, sys, itertools
from collections import Counter
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)
from generador_cobertura_ml import load_data

draws = load_data()
trad = [d for d in draws if d['modalidad'] == 'Tradicional']

# Look at winning draws in Tradicional:
# What is the rank distribution of winning numbers in our Borda model?
print(f"Total sorteos analizados: {len(trad)}")

