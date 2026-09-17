import csv
import json
import numpy as np
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generador_cobertura_ml import load_data

def run_pool_size_backtest():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    if not draws:
        print("Error al cargar datos.")
        return

    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]
    
    print(f"Iniciando Backtesting de Selección de Tamaño de Pool sobre {len(eval_draws)} sorteos pasados...")
    
    pool_sizes = [8, 9, 10, 11, 12, 13, 14, 15]
    results = {k: {'hits_pool': [], 'density': [], 'hits_3plus_pool': 0, 'hits_4plus_pool': 0} for k in pool_sizes}
    
    for i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        target_nums = set(target_draw['numbers'])
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo or (d['sorteo'] == target_sorteo and d['modalidad'] != target_draw['modalidad'])]
        
        if len(train_draws) < 100:
            continue
            
        counts = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']:
                counts[n] += 1
                
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']:
                recency_scores[n] += weight
                
        final_scores = np.zeros(46)
        for n in range(46):
            final_scores[n] = counts[n] * 0.4 + recency_scores[n] * 0.6
            
        ranking = np.argsort(final_scores)[::-1]
        
        for k in pool_sizes:
            top_k = set(ranking[:k])
            hits = len(top_k.intersection(target_nums))
            results[k]['hits_pool'].append(hits)
            results[k]['density'].append(hits / k)
            if hits >= 3:
                results[k]['hits_3plus_pool'] += 1
            if hits >= 4:
                results[k]['hits_4plus_pool'] += 1

    print("\n" + "="*75)
    print(f"{'Pool Size (K)':<15} | {'Avg Hits Pool':<15} | {'Hit Density (Hits/K)':<20} | {'% Pool >= 3 Hits':<18} | {'% Pool >= 4 Hits':<18}")
    print("="*75)
    
    for k in pool_sizes:
        avg_hits = np.mean(results[k]['hits_pool'])
        avg_density = np.mean(results[k]['density']) * 100
        pct_3plus = (results[k]['hits_3plus_pool'] / len(eval_draws)) * 100
        pct_4plus = (results[k]['hits_4plus_pool'] / len(eval_draws)) * 100
        print(f"{k:<15} | {avg_hits:<15.2f} | {avg_density:<19.2f}% | {pct_3plus:<17.1f}% | {pct_4plus:<17.1f}%")
    print("="*75)

if __name__ == '__main__':
    run_pool_size_backtest()
