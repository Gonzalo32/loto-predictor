import csv
import json
import numpy as np
import itertools
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generador_cobertura_ml import (
    load_data, extract_features_vectorized, compute_attention_mlp_scores,
    compute_cooccurrence_matrix, select_tiered_clique_tickets, es_ticket_valido,
    clf_xgb, clf_lgb, clf_cat, clf_rf, clf_mlp, clf_meta, StandardScaler
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
            
        # Entrenar ensemble ML Stacking
        X_train, Y_train = extract_features_vectorized(train_draws)
        
        preds_train = []
        preds_last = []
        
        X_last = X_train[-1:]
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_last_scaled = scaler.transform(X_last)
        
        probs_n = np.zeros(46)
        
        for n in range(46):
            y_n = Y_train[:, n]
            if len(np.unique(y_n)) < 2:
                probs_n[n] = 0.05
                continue
            try:
                clf_xgb.fit(X_train_scaled, y_n)
                p_xgb = clf_xgb.predict_proba(X_last_scaled)[0, 1]
                clf_lgb.fit(X_train_scaled, y_n)
                p_lgb = clf_lgb.predict_proba(X_last_scaled)[0, 1]
                clf_cat.fit(X_train_scaled, y_n)
                p_cat = clf_cat.predict_proba(X_last_scaled)[0, 1]
                probs_n[n] = (p_xgb * 0.35 + p_lgb * 0.35 + p_cat * 0.30)
            except Exception:
                probs_n[n] = 0.05
                
        attn_scores = compute_attention_mlp_scores(train_draws)
        final_scores = probs_n * 0.65 + attn_scores * 0.35
        
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
