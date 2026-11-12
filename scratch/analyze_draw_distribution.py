import os, sys
from collections import Counter
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

# Let's run a quick backtest to get the exact histogram of max hits per draw across 600 draws
from generador_cobertura_ml import load_data
draws = load_data()
trad = [d for d in draws if d['modalidad'] == 'Tradicional']

# In our backtest of 600 draws:
# Trad 3+ occurred in ~40-44 draws (6.7% - 7.3%)
# Trad 2 occurred in how many?
# Trad 1 occurred in how many?
# Trad 0 occurred in how many?

