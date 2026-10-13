import itertools
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generador_cobertura_ml import (
    load_data, compute_exponential_cooccurrence_scores,
    compute_delta_delay_momentum_scores, es_ticket_valido
)
from cobertura_combinatoria import select_wheel_coverage_tickets

def run_experiment():
    draws = load_data('historico_quini_completo.csv')
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-100:]

    print(f"Evaluando {len(eval_draws)} sorteos pasados...")

    strategies = {
        '1. Actual (Overlap Libre)': {'hits_max': [], 'hits_3plus': 0, 'hits_4plus': 0, 'pozo_extra_max': [], 'pozo_3plus': 0},
        '2. Overlap Max <= 3':       {'hits_max': [], 'hits_3plus': 0, 'hits_4plus': 0, 'pozo_extra_max': [], 'pozo_3plus': 0},
        '3. Overlap Max <= 2':       {'hits_max': [], 'hits_3plus': 0, 'hits_4plus': 0, 'pozo_extra_max': [], 'pozo_3plus': 0},
        '4. Wheeling (Triples)':     {'hits_max': [], 'hits_3plus': 0, 'hits_4plus': 0, 'pozo_extra_max': [], 'pozo_3plus': 0},
        '5. Core-Diversificado':    {'hits_max': [], 'hits_3plus': 0, 'hits_4plus': 0, 'pozo_extra_max': [], 'pozo_3plus': 0},
    }

    for target_draw in eval_draws:
        target_sorteo = target_draw['sorteo']
        draw_group = [d for d in draws if d['sorteo'] == target_sorteo]
        mod_dict = {d['modalidad']: set(d['numbers']) for d in draw_group}

        target_trad = mod_dict.get('Tradicional', set())
        pozo_extra = set().union(*[mod_dict.get(m, set()) for m in ['Tradicional', 'Segunda', 'Revancha']])

        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        if len(train_draws) < 80:
            continue

        # 1. Puntuaciones Ensamble
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1

        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency_scores[n] += weight

        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)

        borda = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, 2.0), (pcr_scores, 1.4), (ddm_scores, 1.0)]:
            ranking_m = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(ranking_m):
                borda[n] += w * (45 - rank)

        borda_ranking = [n for n, score in sorted(borda.items(), key=lambda x: x[1], reverse=True)]
        top_pool_14 = borda_ranking[:14]
        top_set_14 = set(top_pool_14)

        combos_14 = list(itertools.combinations(top_pool_14, 6))
        viables_14 = [c for c in combos_14 if es_ticket_valido(c)] or combos_14[:100]

        # --- ESTRATEGIA 1: ACTUAL (OVERLAP LIBRE) ---
        core_pool = set(borda_ranking[:8])
        surge_pool = top_set_14 - core_pool
        v_core = [c for c in viables_14 if sum(1 for n in c if n in core_pool) >= 4] or viables_14
        tA = max(v_core, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
        vB = [c for c in viables_14 if set(c) != set(tA)] or viables_14
        tB = max(vB, key=lambda b: sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))
        borda_C = borda.copy()
        for n in set(tA).union(set(tB)): borda_C[n] *= 0.5
        vC = [c for c in viables_14 if set(c) != set(tA) and set(c) != set(tB)] or viables_14
        tC = max(vC, key=lambda b: sum(borda_C[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
        strat_1_tickets = [set(tA), set(tB), set(tC)]

        # --- ESTRATEGIA 2: OVERLAP MAX <= 3 ---
        t2_A = tA
        v2_B = [c for c in viables_14 if len(set(c).intersection(set(t2_A))) <= 3] or [c for c in viables_14 if set(c) != set(t2_A)]
        t2_B = max(v2_B, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
        v2_C = [c for c in viables_14 if len(set(c).intersection(set(t2_A))) <= 3 and len(set(c).intersection(set(t2_B))) <= 3] or viables_14
        t2_C = max(v2_C, key=lambda b: sum(borda[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
        strat_2_tickets = [set(t2_A), set(t2_B), set(t2_C)]

        # --- ESTRATEGIA 3: OVERLAP MAX <= 2 ---
        t3_A = tA
        v3_B = [c for c in viables_14 if len(set(c).intersection(set(t3_A))) <= 2] or v2_B
        t3_B = max(v3_B, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
        v3_C = [c for c in viables_14 if len(set(c).intersection(set(t3_A))) <= 2 and len(set(c).intersection(set(t3_B))) <= 2] or v2_C
        t3_C = max(v3_C, key=lambda b: sum(borda[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
        strat_3_tickets = [set(t3_A), set(t3_B), set(t3_C)]

        # --- ESTRATEGIA 4: WHEELING (TRIPLES) ---
        scores_w = {n: borda[n] for n in top_pool_14}
        wheel_t = select_wheel_coverage_tickets(viables_14, scores_w, n_tickets=3, target_subsize=3)
        strat_4_tickets = [set(t) for t in wheel_t]

        # --- ESTRATEGIA 5: CORE-DIVERSIFICADO (Partición Disjunta Core/Surge) ---
        # Boleto 1: Top 6 del Core
        # Boleto 2: Alternativo Core + Surge medio (overlap <= 2 con B1)
        # Boleto 3: Surge + Rezagados del Pool (overlap <= 2 con B1 y B2)
        strat_5_tickets = strat_2_tickets  # refined later if needed

        all_strats = {
            '1. Actual (Overlap Libre)': strat_1_tickets,
            '2. Overlap Max <= 3': strat_2_tickets,
            '3. Overlap Max <= 2': strat_3_tickets,
            '4. Wheeling (Triples)': strat_4_tickets,
        }

        for s_name, t_list in all_strats.items():
            max_h = max(len(t.intersection(target_trad)) for t in t_list)
            strategies[s_name]['hits_max'].append(max_h)
            if max_h >= 3: strategies[s_name]['hits_3plus'] += 1
            if max_h >= 4: strategies[s_name]['hits_4plus'] += 1

            if pozo_extra:
                max_pe = max(len(t.intersection(pozo_extra)) for t in t_list)
                strategies[s_name]['pozo_extra_max'].append(max_pe)
                if max_pe >= 3: strategies[s_name]['pozo_3plus'] += 1

    total_eval = len(eval_draws)
    print("\n" + "="*80)
    print("      RESULTADOS EXPERIMENTO: COMPARATIVA DE ESTRATEGIAS (100 SORTEOS)")
    print("="*80)
    print(f"{'Estrategia':<28} | {'Prom Max':<9} | {'% >=3 Trad':<12} | {'% >=4 Trad':<12} | {'% >=3 Pozo Extra':<16}")
    print("-" * 80)
    for s_name, res in all_strats.items():
        data = strategies[s_name]
        mean_h = np.mean(data['hits_max'])
        pct_3 = (data['hits_3plus'] / total_eval) * 100
        pct_4 = (data['hits_4plus'] / total_eval) * 100
        pct_pe = (data['pozo_3plus'] / total_eval) * 100 if data['pozo_extra_max'] else 0.0
        print(f"{s_name:<28} | {mean_h:<9.2f} | {pct_3:<12.1f}% | {pct_4:<12.1f}% | {pct_pe:<16.1f}%")
    print("="*80)

if __name__ == '__main__':
    run_experiment()
