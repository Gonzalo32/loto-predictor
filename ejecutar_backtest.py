import os
import sys

# Redirige a la suite oficial de backtesting comparativo actualizada
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_oficial_comparativo import run_official_backtest

if __name__ == '__main__':
    run_official_backtest()
