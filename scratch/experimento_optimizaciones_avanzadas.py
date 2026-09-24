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

def compute_adaptive_pool(borda_scores, alpha=0.25, min_pool=8, max_pool=12):
    """Optimización A: Pool dinámico ajustado por la desviación estándar de las puntuaciones."""
    scores = np.array([borda_scores[n] for n in range(46)])
    mean_s = np.mean(scores)
    std_s = np.std(scores)
    threshold = mean_s + alpha * std_s
    
    ranked = list(np.argsort(scores)[::-1])
    pool = [n for n in ranked if scores[n] >= threshold]
    
    if len(pool) < min_pool:
        pool = ranked[:min_pool]
    elif len(pool) > max_pool:
        pool = ranked[:max_pool]
    return pool

def compute_cross_modal_transitions(train_draws):
    """Optimización B: Matriz de transición entre modalidades cruzadas (Tradicional -> Segunda -> Revancha)."""
    trans_matrix = np.zeros((46, 46))
    for i in range(1, len(train_draws)):
        prev_nums = train_draws[i-1]['numbers']
        curr_nums = train_draws[i]['numbers']
        for u in prev_nums:
            for v in curr_nums:
                trans_matrix[u][v] += 1.0
    
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return trans_matrix / row_sums

def select_max_coverage_hypergraph_tickets(viables, final_scores, n_tickets=3, max_overlap=3):
    """Optimización C: Selección de 3 tickets hipergrafo que maximizan la cobertura con solapamiento controlado."""
    scored = [(sum(final_scores[n] for n in t), t) for t in viables]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    if not scored:
        return []
    
    selected = [list(scored[0][1])]
    for _, t in scored[1:]:
        if len(selected) >= n_tickets:
            break
        # Control estricto de solapamiento (máximo 'max_overlap' números en común con los seleccionados)
        if all(len(set(t).intersection(set(s))) <= max_overlap for s in selected):
            selected.append(list(t))
            
    # Si faltan tickets por restricciones muy estrictas, relajar a max_overlap + 1
    if len(selected) < n_tickets:
        for _, t in scored[1:]:
            if len(selected) >= n_tickets:
                break
            if t not in selected and all(len(set(t).intersection(set(s))) <= max_overlap + 1 for s in selected):
                selected.append(list(t))
                
    while len(selected) < n_tickets and len(scored) > len(selected):
        selected.append(list(scored[len(selected)][1]))
    return selected

def run_optimization_experiment():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    if not draws:
        print("Error al cargar datos.")
        return

    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-35:]
    
    print("="*90)
    print("EVALUACIÓN DE OPTIMIZACIONES DE MODELO EN 35 SORTEOS HISTÓRICOS PASADOS")
    print("="*90)
    
    # Modelos a comparar:
    # 1. Base (Pool fijo 10 + Tiered Clique)
    # 2. Opt A (Pool Dinámico Adaptativo 8-12 + Tiered Clique)
    # 3. Opt B (Cross-Modal Transitions + Pool 10 + Tiered Clique)
    # 4. Opt C (Pool 10 + Max Coverage Hypergraph Tickets)
    # 5. SUPER-MODELO COMBINADO (Pool Dinámico Adaptativo + Cross-Modal Transitions + Max Coverage Hypergraph)
    
    configs = [
        '1. Modelo Base (Pool 10 + Tiered Clique)',
        '2. Opt A (Pool Dinámico Adaptativo 8-12)',
        '3. Opt B (Cross-Modal Transitions)',
        '4. Opt C (Max Coverage Hypergraph Tickets)',
        '5. SUPER-MODELO COMBINADO (Recomendado)'
    ]
    
    results = {cfg: {'hits_pool': [], 'hits_max_ticket': [], '3plus': 0, '4plus': 0, '5plus': 0} for cfg in configs}
    
    start_time = time.time()
    
    for target_draw in eval_draws:
        target_sorteo = target_draw['sorteo']
        target_nums = set(target_draw['numbers'])
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo or (d['sorteo'] == target_sorteo and d['modalidad'] != target_draw['modalidad'])]
        if len(train_draws) < 120:
            continue
            
        # Puntuaciones base
        counts = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']:
                counts[n] += 1
        freq_scores = np.array([counts[n] for n in range(46)], dtype=float)
        
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 9.2) # Lambda decay optimizado
            for n in d['numbers']:
                recency_scores[n] += weight
                
        attn_scores = compute_attention_mlp_scores(train_draws)
        
        # Cross-modal transitions (Opt B & 5)
        cross_modal_mat = compute_cross_modal_transitions(train_draws[-60:])
        last_nums = train_draws[-1]['numbers']
        cross_scores = np.zeros(46)
        for u in last_nums:
            cross_scores += cross_modal_mat[u]
            
        # Final scores base vs cross-modal
        final_scores_base = freq_scores * 0.30 + recency_scores * 0.40 + attn_scores * 0.30
        final_scores_cross = freq_scores * 0.25 + recency_scores * 0.35 + attn_scores * 0.25 + cross_scores * 0.15
        
        cooccur_matrix = compute_cooccurrence_matrix(train_draws)
        
        # --- Config 1: Base ---
        borda_base = list(np.argsort(final_scores_base)[::-1])
        pool_1 = borda_base[:10]
        combos_1 = [c for c in itertools.combinations(sorted(pool_1), 6) if es_ticket_valido(c)] or list(itertools.combinations(sorted(pool_1), 6))[:50]
        tickets_1 = select_tiered_clique_tickets(combos_1, final_scores_base, cooccur_matrix)
        
        # --- Config 2: Opt A (Pool Dinámico) ---
        pool_2 = compute_adaptive_pool(final_scores_base, alpha=0.25, min_pool=8, max_pool=12)
        combos_2 = [c for c in itertools.combinations(sorted(pool_2), 6) if es_ticket_valido(c)] or list(itertools.combinations(sorted(pool_2), 6))[:50]
        tickets_2 = select_tiered_clique_tickets(combos_2, final_scores_base, cooccur_matrix)
        
        # --- Config 3: Opt B (Cross-Modal) ---
        borda_cross = list(np.argsort(final_scores_cross)[::-1])
        pool_3 = borda_cross[:10]
        combos_3 = [c for c in itertools.combinations(sorted(pool_3), 6) if es_ticket_valido(c)] or list(itertools.combinations(sorted(pool_3), 6))[:50]
        tickets_3 = select_tiered_clique_tickets(combos_3, final_scores_cross, cooccur_matrix)
        
        # --- Config 4: Opt C (Max Coverage Hypergraph) ---
        pool_4 = borda_base[:10]
        combos_4 = [c for c in itertools.combinations(sorted(pool_4), 6) if es_ticket_valido(c)] or list(itertools.combinations(sorted(pool_4), 6))[:50]
        tickets_4 = select_max_coverage_hypergraph_tickets(combos_4, final_scores_base, n_tickets=3, max_overlap=3)
        
    # --- Config 5: SUPER-MODELO COMBINADO (Pool Dinamico + Cross Modal + Tiered Clique) ---
        pool_5 = compute_adaptive_pool(final_scores_cross, alpha=0.20, min_pool=9, max_pool=12)
        combos_5 = [c for c in itertools.combinations(sorted(pool_5), 6) if es_ticket_valido(c)] or list(itertools.combinations(sorted(pool_5), 6))[:50]
        tickets_5 = select_tiered_clique_tickets(combos_5, final_scores_cross, cooccur_matrix)
        
        # Evaluar
        all_pools = [pool_1, pool_2, pool_3, pool_4, pool_5]
        all_tickets = [tickets_1, tickets_2, tickets_3, tickets_4, tickets_5]
        
        for idx, cfg in enumerate(configs):
            p = all_pools[idx]
            t_list = all_tickets[idx]
            
            hits_p = len(set(p).intersection(target_nums))
            max_t = max(len(set(t).intersection(target_nums)) for t in t_list) if t_list else 0
            
            results[cfg]['hits_pool'].append(hits_p)
            results[cfg]['hits_max_ticket'].append(max_t)
            if max_t >= 3: results[cfg]['3plus'] += 1
            if max_t >= 4: results[cfg]['4plus'] += 1
            if max_t >= 5: results[cfg]['5plus'] += 1

    elapsed = time.time() - start_time
    print(f"\nBacktest de optimización completado en {elapsed:.2f} segundos.\n")
    
    print("="*95)
    print(f"{'Configuración / Optimización':<45} | {'Avg Pool Hits':<14} | {'Avg Ticket Hits':<16} | {'3+ Hits %':<10} | {'4+ Hits %':<10}")
    print("="*95)
    
    for cfg in configs:
        h_pool = np.mean(results[cfg]['hits_pool'])
        h_t = np.mean(results[cfg]['hits_max_ticket'])
        n_tot = len(results[cfg]['hits_max_ticket'])
        p3 = (results[cfg]['3plus'] / n_tot) * 100
        p4 = (results[cfg]['4plus'] / n_tot) * 100
        print(f"{cfg:<45} | {h_pool:<14.2f} | {h_t:<16.2f} | {p3:<9.1f}% | {p4:<9.1f}%")
    print("="*95)

if __name__ == '__main__':
    run_optimization_experiment()
