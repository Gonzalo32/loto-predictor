import csv
import json
import numpy as np
import itertools
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generador_cobertura_ml import (
    load_data, compute_attention_mlp_scores,
    compute_cooccurrence_matrix, select_tiered_clique_tickets, es_ticket_valido
)

def run_ml_ensemble_pool_backtest():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    if not draws:
        print("Error al cargar datos.")
        return

    # Evaluaremos en los últimos 20 sorteos para ver rendimiento exacto del ensemble ML
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-20:]
    
    print(f"Iniciando Backtesting del Stacking ML Ensemble sobre {len(eval_draws)} sorteos pasados...")
    
    pool_sizes = [10, 12, 15]
    results = {k: {'hits_pool': [], 'hits_max_ticket': []} for k in pool_sizes}
    
    for i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        target_nums = set(target_draw['numbers'])
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo or (d['sorteo'] == target_sorteo and d['modalidad'] != target_draw['modalidad'])]
        
        if len(train_draws) < 150:
            continue
            
        counts = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']:
                counts[n] += 1
                
        freq_scores = np.zeros(46)
        for n in range(46):
            freq_scores[n] = counts[n]
                
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']:
                recency_scores[n] += weight
                
        attn_scores = compute_attention_mlp_scores(train_draws)
        final_scores = freq_scores * 0.3 + recency_scores * 0.4 + attn_scores * 0.3
        
        borda_ranking = list(np.argsort(final_scores)[::-1])
        cooccur_matrix = compute_cooccurrence_matrix(train_draws)
        
        for k in pool_sizes:
            top_pool = borda_ranking[:k]
            hits = len(set(top_pool).intersection(target_nums))
            results[k]['hits_pool'].append(hits)
            
            # Generar combinaciones y seleccionar 3 boletos
            combos = list(itertools.combinations(sorted(top_pool), 6))
            viables = [c for c in combos if es_ticket_valido(c)]
            if not viables:
                viables = combos[:50]
                
            tickets = select_tiered_clique_tickets(viables, final_scores, cooccur_matrix)
            max_ticket_hits = max(len(set(t).intersection(target_nums)) for t in tickets)
            results[k]['hits_max_ticket'].append(max_ticket_hits)

    print("\n" + "="*70)
    print(f"{'Pool Size (K)':<15} | {'Avg Hits Pool':<15} | {'Avg Max Ticket Hits':<20} | {'3+ Hits Ticket Pct':<20}")
    print("="*70)
    
    for k in pool_sizes:
        avg_h = np.mean(results[k]['hits_pool'])
        avg_t = np.mean(results[k]['hits_max_ticket'])
        pct_3plus_t = np.mean([1 if x >= 3 else 0 for x in results[k]['hits_max_ticket']]) * 100
        print(f"{k:<15} | {avg_h:<15.2f} | {avg_t:<20.2f} | {pct_3plus_t:<19.1f}%")
    print("="*70)

if __name__ == '__main__':
    run_ml_ensemble_pool_backtest()
