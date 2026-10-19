import csv
import json
import itertools
import numpy as np
from collections import defaultdict, Counter
import os
import sys

sys.path.insert(0, 'c:/Users/Administrador/Desktop/lot')
from generador_cobertura_ml import (
    load_data, es_ticket_valido,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_attention_mlp_scores, compute_adaptive_pool
)

def run_diagnostic():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    
    # Train strictly up to 3416 (the state before yesterday's draw 3417)
    train_draws = [d for d in draws if d['sorteo'] <= 3416]
    real_3417 = [d for d in draws if d['sorteo'] == 3417]
    
    real_results = {}
    for d in real_3417:
        real_results[d['modalidad']] = d['numbers']
    pozo_extra = set(real_results['Tradicional'] + real_results['Segunda'] + real_results['Revancha'])
    
    print("="*70)
    print(" DIAGNÓSTICO PROFUNDO: SORTEO #3417 (AYER 14/10/2026)")
    print("="*70)
    print("Resultados Reales:")
    for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
        print(f"  - {mod:12s}: {real_results.get(mod, [])}")
    print(f"  - Pozo Extra  : {sorted(list(pozo_extra))}")
    
    # Calculate scores before 3417
    counts_60 = Counter()
    for d in train_draws[-60:]:
        for n in d['numbers']:
            counts_60[n] += 1
            
    recency_scores = np.zeros(46)
    for t_idx, d in enumerate(train_draws[-30:]):
        weight = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']:
            recency_scores[n] += weight
            
    pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
    ddm_scores = compute_delta_delay_momentum_scores(train_draws)
    attn_scores = compute_attention_mlp_scores(train_draws)
    
    # Borda ensemble
    borda = defaultdict(float)
    models_to_rank = [
        (counts_60, 1.5),
        (recency_scores, 2.0),
        (pcr_scores, 1.4),
        (ddm_scores, 1.8),
        ({n: attn_scores[n] for n in range(46)}, 1.3)
    ]
    for m_scores, w in models_to_rank:
        ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
        for rank, n in enumerate(ranked):
            borda[n] += w * (45 - rank)
            
    borda_ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)
    
    print("\n[TOP 18 NÚMEROS DEL RANKING BORDA ANTES DEL SORTEO]:")
    for r, n in enumerate(borda_ranking[:18]):
        in_trad = 'TRAD' if n in real_results['Tradicional'] else ''
        in_seg = 'SEG' if n in real_results['Segunda'] else ''
        in_rev = 'REV' if n in real_results['Revancha'] else ''
        in_ss = 'SS' if n in real_results['SiempreSale'] else ''
        matches = " ".join(filter(None, [in_trad, in_seg, in_rev, in_ss]))
        print(f"  Pos #{r+1:02d} | Nro {n:02d} | Score: {borda[n]:.1f} | {matches}")
        
    top_14 = sorted(borda_ranking[:14])
    top_12 = sorted(borda_ranking[:12])
    top_10 = sorted(borda_ranking[:10])
    
    print(f"\nPool 14: {top_14} -> Hits Pozo Extra: {len(set(top_14).intersection(pozo_extra))}/16")
    print(f"Pool 12: {top_12} -> Hits Pozo Extra: {len(set(top_12).intersection(pozo_extra))}/16")
    print(f"Pool 10: {top_10} -> Hits Pozo Extra: {len(set(top_10).intersection(pozo_extra))}/16")

    # Evaluate different ticket generation methods on top_14, top_12, top_10
    def evaluate_strategy(name, tickets):
        print(f"\n>>> Estrategia: {name}")
        for idx, t in enumerate(tickets):
            t_set = set(t)
            hits_trad = len(t_set.intersection(set(real_results['Tradicional'])))
            hits_seg = len(t_set.intersection(set(real_results['Segunda'])))
            hits_rev = len(t_set.intersection(set(real_results['Revancha'])))
            hits_ss = len(t_set.intersection(set(real_results['SiempreSale'])))
            hits_pe = len(t_set.intersection(pozo_extra))
            print(f"  T{idx+1}: {sorted(t)} | Trad: {hits_trad} | Seg: {hits_seg} | Rev: {hits_rev} | SS: {hits_ss} | PozoExtra: {hits_pe}")

    # Método actual (el que se usó):
    core_pool = set(borda_ranking[:8])
    surge_pool = set(top_14) - core_pool
    viables_14 = [c for c in itertools.combinations(top_14, 6) if es_ticket_valido(c)]
    
    viables_core = [c for c in viables_14 if sum(1 for n in c if n in core_pool) >= 4] or viables_14
    ticket_A = max(viables_core, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
    
    viables_B = [c for c in viables_14 if len(set(c).intersection(set(ticket_A))) <= 3] or viables_14
    ticket_B = max(viables_B, key=lambda b: sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))
    
    borda_C = borda.copy()
    for n in set(ticket_A).union(set(ticket_B)): borda_C[n] *= 0.5
    viables_C = [c for c in viables_14 if len(set(c).intersection(set(ticket_A))) <= 3 and len(set(c).intersection(set(ticket_B))) <= 3] or viables_14
    ticket_C = max(viables_C, key=lambda b: sum(borda_C[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
    
    evaluate_strategy("1. Método Actual (Core-Surge con Solapamiento Débil)", [ticket_A, ticket_B, ticket_C])

    # Nueva Estrategia A: Sin SPOF (ningún número puede repetirse en 3 tickets) + Cobertura de Cliques
    def select_no_spof_cliques(viables, pool):
        # Seleccionar ticket A por afinidad pura
        t_A = max(viables, key=lambda b: sum(borda[n] for n in b) + 300.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
        # Ticket B: overlap <= 3 con A
        v_B = [c for c in viables if len(set(c).intersection(set(t_A))) <= 3]
        t_B = max(v_B, key=lambda b: sum(borda[n] for n in b) + 400.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
        
        # Ticket C: overlap <= 3 con A y con B, Y además NO puede contener números que ya estén en A Y B (intersección A y B no permitida en C)
        overlap_AB = set(t_A).intersection(set(t_B))
        v_C = [c for c in viables if len(set(c).intersection(set(t_A))) <= 3 and len(set(c).intersection(set(t_B))) <= 3 and not any(n in overlap_AB for n in c)]
        if not v_C:
            v_C = [c for c in viables if len(set(c).intersection(set(t_A))) <= 3 and len(set(c).intersection(set(t_B))) <= 3]
        
        # Priorizar en C números del pool que aún no hayan sido cubiertos
        covered_AB = set(t_A).union(set(t_B))
        t_C = max(v_C, key=lambda b: sum(borda[n] for n in b) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) + 120.0 * sum(1 for n in b if n not in covered_AB))
        return [t_A, t_B, t_C]
        
    evaluate_strategy("2. Anti-SPOF (Cap Máximo 2 Repeticiones por Número)", select_no_spof_cliques(viables_14, top_14))

    # Nueva Estrategia B: Pool Compacto 11 números (Mayor densidad combinatoria)
    viables_11 = [c for c in itertools.combinations(borda_ranking[:11], 6) if es_ticket_valido(c)]
    evaluate_strategy("3. Pool Compacto Top 11 + Anti-SPOF", select_no_spof_cliques(viables_11, borda_ranking[:11]))

    # Nueva Estrategia C: Cluster Co-Occurrence Packing (Empaquetador de Afinidad)
    # En lugar de imponer restricciones arbitrarias de 'core >= 4', buscamos empaquetar números con alta co-ocurrencia histórica mutua
    def select_cooccurrence_clusters(viables):
        scored_combos = []
        for c in viables:
            # Score compuesto: Borda individual + co-ocurrencia cuadrática de pares
            s_borda = sum(borda[n] for n in c)
            s_cooccur = sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
            scored_combos.append((s_borda + 400.0 * s_cooccur, c))
        scored_combos.sort(key=lambda x: x[0], reverse=True)
        
        # Selección greedy de 3 tickets maximizando cobertura con overlap <= 3 y sin repetición triple
        t1 = scored_combos[0][1]
        t2 = None
        for _, c in scored_combos[1:]:
            if len(set(c).intersection(set(t1))) <= 3:
                t2 = c
                break
        if t2 is None: t2 = scored_combos[1][1]
        
        overlap_12 = set(t1).intersection(set(t2))
        t3 = None
        for _, c in scored_combos[1:]:
            if len(set(c).intersection(set(t1))) <= 3 and len(set(c).intersection(set(t2))) <= 3:
                if not any(n in overlap_12 for n in c):
                    t3 = c
                    break
        if t3 is None:
            for _, c in scored_combos[1:]:
                if len(set(c).intersection(set(t1))) <= 3 and len(set(c).intersection(set(t2))) <= 3:
                    t3 = c
                    break
        return [t1, t2, t3]

    evaluate_strategy("4. Pure Co-Occurrence Cluster Packing (Pool 14)", select_cooccurrence_clusters(viables_14))
    evaluate_strategy("5. Pure Co-Occurrence Cluster Packing (Pool 11)", select_cooccurrence_clusters(viables_11))

if __name__ == '__main__':
    run_diagnostic()
