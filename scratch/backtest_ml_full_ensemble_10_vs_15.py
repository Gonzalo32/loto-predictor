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

def run_full_ml_ensemble_backtest():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    if not draws:
        print("Error al cargar datos.")
        return

    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-30:]
    
    print("="*80)
    print(f"BACKTESTING ML ENSEMBLE COMPLETO (STACKING + DEEP ATTENTION): POOL 15 VS POOL 10")
    print(f"Evaluación estricta sobre {len(eval_draws)} sorteos pasados cronológicos...")
    print("="*80)
    
    p15_hits_pool = []
    p15_hits_ticket = []
    p15_ticket_3plus = 0
    p15_ticket_4plus = 0
    
    p10_hits_pool = []
    p10_hits_ticket = []
    p10_ticket_3plus = 0
    p10_ticket_4plus = 0
    
    start_time = time.time()
    
    for i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        target_nums = set(target_draw['numbers'])
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo or (d['sorteo'] == target_sorteo and d['modalidad'] != target_draw['modalidad'])]
        
        if len(train_draws) < 120:
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
        
        # --- POOL 15 ---
        top15 = borda_ranking[:15]
        hits15_pool = len(set(top15).intersection(target_nums))
        p15_hits_pool.append(hits15_pool)
        
        combos15 = list(itertools.combinations(sorted(top15), 6))
        viables15 = [c for c in combos15 if es_ticket_valido(c)]
        if not viables15: viables15 = combos15[:50]
        tickets15 = select_tiered_clique_tickets(viables15, final_scores, cooccur_matrix)
        max_hits15 = max(len(set(t).intersection(target_nums)) for t in tickets15)
        p15_hits_ticket.append(max_hits15)
        if max_hits15 >= 3: p15_ticket_3plus += 1
        if max_hits15 >= 4: p15_ticket_4plus += 1
        
        # --- POOL 10 (NUEVO MODELO) ---
        top10 = borda_ranking[:10]
        hits10_pool = len(set(top10).intersection(target_nums))
        p10_hits_pool.append(hits10_pool)
        
        combos10 = list(itertools.combinations(sorted(top10), 6))
        viables10 = [c for c in combos10 if es_ticket_valido(c)]
        if not viables10: viables10 = combos10[:50]
        tickets10 = select_tiered_clique_tickets(viables10, final_scores, cooccur_matrix)
        max_hits10 = max(len(set(t).intersection(target_nums)) for t in tickets10)
        p10_hits_ticket.append(max_hits10)
        if max_hits10 >= 3: p10_ticket_3plus += 1
        if max_hits10 >= 4: p10_ticket_4plus += 1
        
        if (i + 1) % 10 == 0:
            print(f"Evaluados {i + 1}/{len(eval_draws)} sorteos pasados...")

    elapsed = time.time() - start_time
    total = len(p10_hits_pool)
    
    print("\n" + "="*80)
    print(f"RESUMEN FINAL DE COMPARA (ML ENSEMBLE - {total} SORTEOS):")
    print("="*80)
    
    print(f"\nPORCENTAJE DE SORTEOS CON BOLETAS GANADORAS DE 3+ ACIERTOS:")
    print(f"  • POOL 15 ANTERIOR: {(p15_ticket_3plus/total)*100:.1f}%")
    print(f"  • POOL 10 NUEVO REDUCIDO: {(p10_ticket_3plus/total)*100:.1f}%")
    print(f"  --> AUMENTO DE EFECTIVIDAD: +{((p10_ticket_3plus/total) - (p15_ticket_3plus/total))*100:+.1f} PUNTOS PORCENTUALES")
    
    print(f"\nPORCENTAJE DE SORTEOS CON BOLETAS GANADORAS DE 4+ ACIERTOS:")
    print(f"  • POOL 15 ANTERIOR: {(p15_ticket_4plus/total)*100:.1f}%")
    print(f"  • POOL 10 NUEVO REDUCIDO: {(p10_ticket_4plus/total)*100:.1f}%")
    print(f"  --> AUMENTO DE EFECTIVIDAD: +{((p10_ticket_4plus/total) - (p15_ticket_4plus/total))*100:+.1f} PUNTOS PORCENTUALES")
    
    print(f"\nDENSIDAD DE PRECISIÓN DEL POOL (Aciertos por cada número sugerido):")
    print(f"  • POOL 15 ANTERIOR: {(np.mean(p15_hits_pool)/15)*100:.2f}% de aciertos por número")
    print(f"  • POOL 10 NUEVO REDUCIDO: {(np.mean(p10_hits_pool)/10)*100:.2f}% de aciertos por número")
    print("="*80)

if __name__ == '__main__':
    run_full_ml_ensemble_backtest()
