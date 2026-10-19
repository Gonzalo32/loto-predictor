import csv
import json
import itertools
import numpy as np
from collections import defaultdict, Counter
import time
import os
import sys

sys.path.insert(0, 'c:/Users/Administrador/Desktop/lot')
from generador_cobertura_ml import (
    load_data, es_ticket_valido,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_attention_mlp_scores, compute_adaptive_pool
)

def run_packing_backtest():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-40:]
    
    print("="*90)
    print(" EVALUACIÓN COMPARATIVA DE MOTORES DE EMPAQUETADO DE BOLETOS (ÚLTIMOS 40 SORTEOS)")
    print(f" Sorteos #{eval_draws[0]['sorteo']} al #{eval_draws[-1]['sorteo']}")
    print("="*90)
    
    # Métricas acumuladas
    stats = {
        '1. Motor Actual (Core-Surge / Greedy)': {'3plus': 0, '4plus': 0, '5plus': 0, 'pe_5plus': 0, 'pe_6plus': 0, 'max_hits': []},
        '2. Motor Nuevo A (Wheel Pairwise Max-Coverage + Anti-SPOF)': {'3plus': 0, '4plus': 0, '5plus': 0, 'pe_5plus': 0, 'pe_6plus': 0, 'max_hits': []},
        '3. Motor Nuevo B (Multi-Modal Specialized + Anti-SPOF)': {'3plus': 0, '4plus': 0, '5plus': 0, 'pe_5plus': 0, 'pe_6plus': 0, 'max_hits': []},
    }
    
    start_time = time.time()
    
    for idx_step, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        
        # Real results for this draw across modalities
        group = [d for d in draws if d['sorteo'] == target_sorteo]
        real_mods = {d['modalidad']: set(d['numbers']) for d in group}
        target_trad = real_mods.get('Tradicional', set())
        target_seg = real_mods.get('Segunda', set())
        target_rev = real_mods.get('Revancha', set())
        target_ss = real_mods.get('SiempreSale', set())
        pozo_extra = target_trad.union(target_seg).union(target_rev)
        
        # Historical training data up to before this draw
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        if len(train_draws) < 100:
            continue
            
        # Puntuaciones
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1
                
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency_scores[n] += weight
            
        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)
        attn_scores = compute_attention_mlp_scores(train_draws)
        
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
        top_pool = sorted(borda_ranking[:14])
        
        combos = [c for c in itertools.combinations(top_pool, 6) if es_ticket_valido(c)]
        if not combos:
            combos = list(itertools.combinations(top_pool, 6))[:50]
            
        # -------------------------------------------------------------
        # 1. MOTOR ACTUAL (CORE-SURGE / GREEDY)
        # -------------------------------------------------------------
        core_pool = set(borda_ranking[:8])
        surge_pool = set(top_pool) - core_pool
        v_core = [c for c in combos if sum(1 for n in c if n in core_pool) >= 4] or combos
        tA_act = max(v_core, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
        
        v_B = [c for c in combos if len(set(c).intersection(set(tA_act))) <= 3] or combos
        tB_act = max(v_B, key=lambda b: sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))
        
        borda_C = borda.copy()
        for n in set(tA_act).union(set(tB_act)): borda_C[n] *= 0.5
        v_C = [c for c in combos if len(set(c).intersection(set(tA_act))) <= 3 and len(set(c).intersection(set(tB_act))) <= 3] or combos
        tC_act = max(v_C, key=lambda b: sum(borda_C[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
        tickets_actual = [tA_act, tB_act, tC_act]

        # -------------------------------------------------------------
        # 2. MOTOR NUEVO A (WHEEL PAIRWISE MAX-COVERAGE + ANTI-SPOF)
        # -------------------------------------------------------------
        # Pre-calcula puntuación de calidad intrínseca para cada combinación
        scored_combos = []
        for c in combos:
            s_nodes = sum(borda[n] for n in c)
            s_edges = sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
            scored_combos.append((s_nodes + 450.0 * s_edges, c))
        scored_combos.sort(key=lambda x: x[0], reverse=True)
        
        # Selección conjunta optimizada
        t1_A = scored_combos[0][1]
        
        # t2: máximo score con overlap <= 3 con t1
        t2_A = None
        for _, c in scored_combos[1:]:
            if len(set(c).intersection(set(t1_A))) <= 3:
                t2_A = c
                break
        if t2_A is None: t2_A = scored_combos[1][1]
        
        # t3: overlap <= 3 con t1 y t2, Y restricción anti-SPOF (no puede contener la intersección de t1 y t2)
        overlap_12 = set(t1_A).intersection(set(t2_A))
        covered_12 = set(t1_A).union(set(t2_A))
        uncovered = set(top_pool) - covered_12
        
        candidates_t3 = []
        for s, c in scored_combos:
            if c == t1_A or c == t2_A:
                continue
            if len(set(c).intersection(set(t1_A))) <= 3 and len(set(c).intersection(set(t2_A))) <= 3:
                # Penalizar o prohibir contener números de overlap_12 (anti-SPOF estricto)
                if not any(n in overlap_12 for n in c):
                    # Bonificación por cubrir números huérfanos del pool
                    bonus = 180.0 * sum(1 for n in c if n in uncovered)
                    candidates_t3.append((s + bonus, c))
                    
        if candidates_t3:
            candidates_t3.sort(key=lambda x: x[0], reverse=True)
            t3_A = candidates_t3[0][1]
        else:
            t3_A = scored_combos[2][1]
            
        tickets_nuevo_A = [t1_A, t2_A, t3_A]

        # -------------------------------------------------------------
        # 3. MOTOR NUEVO B (MULTI-MODAL SPECIALIZED + ANTI-SPOF)
        # -------------------------------------------------------------
        # Matriz de afinidad por modalidad
        # Ticket 1 (Tradicional): Enfatiza pares de Tradicional
        # Ticket 2 (Segunda): Enfatiza pares de Segunda
        # Ticket 3 (Pozo Extra): Cobertura complementaria
        trad_counts = Counter()
        seg_counts = Counter()
        for d in train_draws[-40:]:
            if d['modalidad'] == 'Tradicional':
                for n in d['numbers']: trad_counts[n] += 1
            elif d['modalidad'] == 'Segunda':
                for n in d['numbers']: seg_counts[n] += 1
                
        # T1 Tradicional
        t1_B = max(combos, key=lambda c: sum(borda[n] + 15.0 * trad_counts[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        
        # T2 Segunda (overlap <= 3 con T1)
        v_seg = [c for c in combos if len(set(c).intersection(set(t1_B))) <= 3] or combos
        t2_B = max(v_seg, key=lambda c: sum(borda[n] + 20.0 * seg_counts[n] for n in c) + 400.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        
        # T3 Pozo Extra (anti-SPOF: no nros repetidos en 1 y 2, maximiza cobertura global)
        overlap_12_B = set(t1_B).intersection(set(t2_B))
        covered_12_B = set(t1_B).union(set(t2_B))
        uncovered_B = set(top_pool) - covered_12_B
        
        v_pe = [c for c in combos if len(set(c).intersection(set(t1_B))) <= 3 and len(set(c).intersection(set(t2_B))) <= 3 and not any(n in overlap_12_B for n in c)]
        if not v_pe:
            v_pe = [c for c in combos if len(set(c).intersection(set(t1_B))) <= 3 and len(set(c).intersection(set(t2_B))) <= 3] or combos
            
        t3_B = max(v_pe, key=lambda c: sum(borda[n] for n in c) + 300.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 200.0 * sum(1 for n in c if n in uncovered_B))
        tickets_nuevo_B = [t1_B, t2_B, t3_B]

        # -------------------------------------------------------------
        # EVALUACIÓN DE LAS 3 ESTRATEGIAS EN EL SORTEO OBJETIVO
        # -------------------------------------------------------------
        engines = [
            ('1. Motor Actual (Core-Surge / Greedy)', tickets_actual),
            ('2. Motor Nuevo A (Wheel Pairwise Max-Coverage + Anti-SPOF)', tickets_nuevo_A),
            ('3. Motor Nuevo B (Multi-Modal Specialized + Anti-SPOF)', tickets_nuevo_B)
        ]
        
        for name, t_list in engines:
            max_hit_modal = 0
            max_hit_pe = 0
            for t in t_list:
                t_set = set(t)
                h_tr = len(t_set.intersection(target_trad))
                h_sg = len(t_set.intersection(target_seg))
                h_rv = len(t_set.intersection(target_rev))
                h_ss = len(t_set.intersection(target_ss))
                h_pe = len(t_set.intersection(pozo_extra))
                
                max_hit_modal = max(max_hit_modal, h_tr, h_sg, h_rv, h_ss)
                max_hit_pe = max(max_hit_pe, h_pe)
                
            stats[name]['max_hits'].append(max_hit_modal)
            if max_hit_modal >= 3: stats[name]['3plus'] += 1
            if max_hit_modal >= 4: stats[name]['4plus'] += 1
            if max_hit_modal >= 5: stats[name]['5plus'] += 1
            if max_hit_pe >= 5: stats[name]['pe_5plus'] += 1
            if max_hit_pe >= 6: stats[name]['pe_6plus'] += 1

    elapsed = time.time() - start_time
    print(f"\nSimulación walk-forward completada en {elapsed:.2f} segundos.\n")
    
    print("="*105)
    print(f"{'Estrategia de Empaquetado':<55} | {'Avg Max Hit':<12} | {'3+ Hits %':<10} | {'4+ Hits (Premios)':<18} | {'Pozo Extra 5+ Hits'}")
    print("="*105)
    
    total_sorteos = len(eval_draws)
    for name in stats:
        avg_h = np.mean(stats[name]['max_hits'])
        p3 = (stats[name]['3plus'] / total_sorteos) * 100
        n4 = stats[name]['4plus']
        p4 = (n4 / total_sorteos) * 100
        pe5 = stats[name]['pe_5plus']
        pe6 = stats[name]['pe_6plus']
        print(f"{name:<55} | {avg_h:<12.2f} | {p3:<9.1f}% | {n4} ({p4:.1f}%)          | {pe5} veces (6 aciertos: {pe6})")
    print("="*105)

if __name__ == '__main__':
    run_packing_backtest()
