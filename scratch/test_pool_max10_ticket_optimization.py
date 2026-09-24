import csv
import json
import numpy as np
import itertools
import os
import sys
from collections import defaultdict, Counter
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["LIGHTGBM_VERBOSE"] = "-1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generador_cobertura_ml import (
    load_data, compute_attention_mlp_scores,
    compute_cooccurrence_matrix, select_tiered_clique_tickets, es_ticket_valido,
    clf_xgb, clf_lgb, clf_cat, clf_rf, clf_mlp, clf_meta, StandardScaler
)

def select_max_score_tickets(viables, final_scores, n_tickets=3):
    """Estrategia 2: Selecciona los N tickets viables con mayor suma de score individual."""
    scored = []
    for t in viables:
        s = sum(final_scores[n] for n in t)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [list(t[1]) for t in scored[:n_tickets]]

def select_cooccur_synergy_tickets(viables, final_scores, cooccur_matrix, n_tickets=3):
    """Estrategia 3: Selecciona los N tickets viables con mayor sinergia de co-ocurrencia histórica."""
    scored = []
    for t in viables:
        score_ml = sum(final_scores[n] for n in t)
        score_co = 0.0
        for u, v in itertools.combinations(t, 2):
            score_co += cooccur_matrix[u][v]
        total_score = score_ml * 0.5 + score_co * 0.5
        scored.append((total_score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Seleccionar tickets diversos
    selected = [list(scored[0][1])]
    for _, t in scored[1:]:
        if len(selected) >= n_tickets:
            break
        # Evitar tickets casi idénticos (más de 4 números repetidos)
        if all(len(set(t).intersection(set(s))) <= 4 for s in selected):
            selected.append(list(t))
    while len(selected) < n_tickets and len(scored) > len(selected):
        selected.append(list(scored[len(selected)][1]))
    return selected

def select_max_diversity_tickets(viables, final_scores, n_tickets=3):
    """Estrategia 4: Maximiza la cobertura del pool minimizando el solapamiento entre los 3 tickets."""
    scored = [(sum(final_scores[n] for n in t), t) for t in viables]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    if not scored:
        return []
    
    selected = [list(scored[0][1])]
    for _, t in scored[1:]:
        if len(selected) >= n_tickets:
            break
        # Exigir menor solapamiento (máximo 3 números compartidos con tickets anteriores)
        if all(len(set(t).intersection(set(s))) <= 3 for s in selected):
            selected.append(list(t))
            
    # Si faltan tickets, completar con los mejores disponibles
    idx = 1
    while len(selected) < n_tickets and idx < len(scored):
        if scored[idx][1] not in selected:
            selected.append(list(scored[idx][1]))
        idx += 1
    return selected


def run_pool_max10_backtest():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    if not draws:
        print("Error al cargar datos.")
        return

    # Evaluaremos en los últimos 30 sorteos para máxima solidez estadística
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-30:]
    
    print("="*85)
    print(f"EXPERIMENTO BACKTEST: POOL RESTREÑIDO (MAX 10) Y COMPARATIVA DE ESTRATEGIAS DE TICKETS")
    print(f"Evaluando {len(eval_draws)} sorteos pasados cronológicos...")
    print("="*85)
    
    pool_sizes = [10, 9, 8]
    strategies = [
        ('Tiered Clique (Actual)', select_tiered_clique_tickets),
        ('Max ML Score', select_max_score_tickets),
        ('Co-occur Synergy', select_cooccur_synergy_tickets),
        ('Max Diversity Cover', select_max_diversity_tickets)
    ]
    
    # Estructura de resultados
    results = {}
    for k in pool_sizes:
        results[k] = {
            'hits_pool': [],
            'strats': {strat_name: {'max_hits': [], 'hits_3plus': 0, 'hits_4plus': 0, 'hits_5plus': 0} for strat_name, _ in strategies}
        }
        
    start_time = time.time()
    
    for i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        target_nums = set(target_draw['numbers'])
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo or (d['sorteo'] == target_sorteo and d['modalidad'] != target_draw['modalidad'])]
        
        if len(train_draws) < 120:
            continue
            
        # Puntuaciones
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
            hits_pool = len(set(top_pool).intersection(target_nums))
            results[k]['hits_pool'].append(hits_pool)
            
            combos = list(itertools.combinations(sorted(top_pool), 6))
            viables = [c for c in combos if es_ticket_valido(c)]
            if not viables:
                viables = combos[:50]
                
            for strat_name, strat_fn in strategies:
                if strat_name == 'Tiered Clique (Actual)':
                    tickets = strat_fn(viables, final_scores, cooccur_matrix)
                elif strat_name == 'Co-occur Synergy':
                    tickets = strat_fn(viables, final_scores, cooccur_matrix)
                else:
                    tickets = strat_fn(viables, final_scores)
                    
                if not tickets:
                    max_h = 0
                else:
                    max_h = max(len(set(t).intersection(target_nums)) for t in tickets)
                    
                results[k]['strats'][strat_name]['max_hits'].append(max_h)
                if max_h >= 3: results[k]['strats'][strat_name]['hits_3plus'] += 1
                if max_h >= 4: results[k]['strats'][strat_name]['hits_4plus'] += 1
                if max_h >= 5: results[k]['strats'][strat_name]['hits_5plus'] += 1
                
    elapsed = time.time() - start_time
    print(f"\nBacktest finalizado en {elapsed:.2f} segundos.\n")
    
    # Imprimir resultados detallados
    for k in pool_sizes:
        avg_pool_hits = np.mean(results[k]['hits_pool'])
        print("="*85)
        print(f"--- RESULTADOS PARA POOL = {k} NÚMEROS (Promedio de aciertos en Pool: {avg_pool_hits:.2f} / 6) ---")
        print("="*85)
        print(f"{'Estrategia de Selección de 3 Tickets':<30} | {'Avg Hits Ticket':<16} | {'3+ Aciertos %':<15} | {'4+ Aciertos %':<15}")
        print("-" * 85)
        
        for strat_name, _ in strategies:
            max_hits_list = results[k]['strats'][strat_name]['max_hits']
            n_eval = len(max_hits_list)
            if n_eval == 0:
                continue
            avg_ticket_hits = np.mean(max_hits_list)
            pct_3plus = (results[k]['strats'][strat_name]['hits_3plus'] / n_eval) * 100
            pct_4plus = (results[k]['strats'][strat_name]['hits_4plus'] / n_eval) * 100
            print(f"{strat_name:<30} | {avg_ticket_hits:<16.2f} | {pct_3plus:<14.1f}% | {pct_4plus:<14.1f}%")
        print("\n")

if __name__ == '__main__':
    run_pool_max10_backtest()
