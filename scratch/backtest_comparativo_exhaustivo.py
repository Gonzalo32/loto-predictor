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

def run_walk_forward_backtest(n_eval_draws=100):
    draws = load_data('historico_quini_completo.csv')
    trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
    eval_draws = trad_draws[-n_eval_draws:]

    print("=" * 95)
    print("      BACKTESTING RIGUROSO WALK-FORWARD: MODELO ANTERIOR VS MODELO NUEVO MEJORADO")
    print(f"      Evaluando {len(eval_draws)} sorteos pasados (Sorteos #{eval_draws[0]['sorteo']} al #{eval_draws[-1]['sorteo']})")
    print("=" * 95)

    stats = {
        'Modelo Anterior (Solapamiento Libre)': {
            'hits_trad': [], 'hits_3plus_trad': 0, 'hits_4plus_trad': 0, 'hits_5plus_trad': 0,
            'hits_pe': [], 'hits_3plus_pe': 0, 'hits_4plus_pe': 0,
            'distinct_numbers_covered': []
        },
        'Modelo Nuevo (Solapamiento Ortogonal <= 3)': {
            'hits_trad': [], 'hits_3plus_trad': 0, 'hits_4plus_trad': 0, 'hits_5plus_trad': 0,
            'hits_pe': [], 'hits_3plus_pe': 0, 'hits_4plus_pe': 0,
            'distinct_numbers_covered': []
        },
        'Azar / Aleatorio (Línea Base)': {
            'hits_trad': [], 'hits_3plus_trad': 0, 'hits_4plus_trad': 0, 'hits_5plus_trad': 0,
            'hits_pe': [], 'hits_3plus_pe': 0, 'hits_4plus_pe': 0,
            'distinct_numbers_covered': []
        }
    }

    start_time = time.time()
    np.random.seed(42)

    for i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        draw_group = [d for d in draws if d['sorteo'] == target_sorteo]
        mod_dict = {d['modalidad']: set(d['numbers']) for d in draw_group}

        target_trad = mod_dict.get('Tradicional', set())
        pozo_extra = set().union(*[mod_dict.get(m, set()) for m in ['Tradicional', 'Segunda', 'Revancha']])

        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        if len(train_draws) < 80:
            continue

        # 1. Ranking Borda Ensamble (Mismo Pool para ambos modelos para aislar el efecto de la estrategia)
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

        # --- MODELO ANTERIOR ---
        core_pool = set(borda_ranking[:8])
        surge_pool = top_set_14 - core_pool
        v_core = [c for c in viables_14 if sum(1 for n in c if n in core_pool) >= 4] or viables_14
        tA_old = max(v_core, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
        vB_old = [c for c in viables_14 if set(c) != set(tA_old)] or viables_14
        tB_old = max(vB_old, key=lambda b: sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))
        borda_C = borda.copy()
        for n in set(tA_old).union(set(tB_old)): borda_C[n] *= 0.5
        vC_old = [c for c in viables_14 if set(c) != set(tA_old) and set(c) != set(tB_old)] or viables_14
        tC_old = max(vC_old, key=lambda b: sum(borda_C[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
        tickets_old = [set(tA_old), set(tB_old), set(tC_old)]

        # --- MODELO NUEVO (SOLAPAMIENTO ORTOGONAL <= 3) ---
        tA_new = tA_old
        vB_new = [c for c in viables_14 if len(set(c).intersection(set(tA_new))) <= 3] or [c for c in viables_14 if set(c) != set(tA_new)]
        tB_new = max(vB_new, key=lambda b: sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))
        borda_C_new = borda.copy()
        for n in set(tA_new).union(set(tB_new)): borda_C_new[n] *= 0.5
        vC_new = [c for c in viables_14 if len(set(c).intersection(set(tA_new))) <= 3 and len(set(c).intersection(set(tB_new))) <= 3] or [c for c in viables_14 if set(c) != set(tA_new) and set(c) != set(tB_new)]
        tC_new = max(vC_new, key=lambda b: sum(borda_C_new[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
        tickets_new = [set(tA_new), set(tB_new), set(tC_new)]

        # --- AZAR (3 BOLETOS ALEATORIOS) ---
        tickets_rnd = [set(np.random.choice(46, 6, replace=False)) for _ in range(3)]

        run_configs = [
            ('Modelo Anterior (Solapamiento Libre)', tickets_old),
            ('Modelo Nuevo (Solapamiento Ortogonal <= 3)', tickets_new),
            ('Azar / Aleatorio (Línea Base)', tickets_rnd)
        ]

        for name, tkts in run_configs:
            # Números únicos cubiertos por los 3 boletos
            covered_unique = len(set.union(*tkts))
            stats[name]['distinct_numbers_covered'].append(covered_unique)

            # Aciertos Tradicional
            max_h_trad = max(len(t.intersection(target_trad)) for t in tkts)
            stats[name]['hits_trad'].append(max_h_trad)
            if max_h_trad >= 3: stats[name]['hits_3plus_trad'] += 1
            if max_h_trad >= 4: stats[name]['hits_4plus_trad'] += 1
            if max_h_trad >= 5: stats[name]['hits_5plus_trad'] += 1

            # Aciertos Pozo Extra
            if pozo_extra:
                max_h_pe = max(len(t.intersection(pozo_extra)) for t in tkts)
                stats[name]['hits_pe'].append(max_h_pe)
                if max_h_pe >= 3: stats[name]['hits_3plus_pe'] += 1
                if max_h_pe >= 4: stats[name]['hits_4plus_pe'] += 1

    elapsed = time.time() - start_time
    total = len(eval_draws)

    print("\n" + "=" * 95)
    print(f"                                RESULTADOS FINALES ({total} SORTEOS)")
    print("=" * 95)
    print(f"{'Métrica Evaluada':<42} | {'Modelo Anterior':<16} | {'Modelo Nuevo':<16} | {'Azar Puro'}")
    print("-" * 95)

    cov_old = np.mean(stats['Modelo Anterior (Solapamiento Libre)']['distinct_numbers_covered'])
    cov_new = np.mean(stats['Modelo Nuevo (Solapamiento Ortogonal <= 3)']['distinct_numbers_covered'])
    cov_rnd = np.mean(stats['Azar / Aleatorio (Línea Base)']['distinct_numbers_covered'])
    print(f"{'Números Únicos Cubiertos (promedio)':<42} | {cov_old:<16.1f} | {cov_new:<16.1f} | {cov_rnd:<.1f}")

    h_old = np.mean(stats['Modelo Anterior (Solapamiento Libre)']['hits_trad'])
    h_new = np.mean(stats['Modelo Nuevo (Solapamiento Ortogonal <= 3)']['hits_trad'])
    h_rnd = np.mean(stats['Azar / Aleatorio (Línea Base)']['hits_trad'])
    print(f"{'Promedio Acierto Máx / Sorteo (Tradicional)':<42} | {h_old:<16.2f} | {h_new:<16.2f} | {h_rnd:<.2f}")

    p3_old = (stats['Modelo Anterior (Solapamiento Libre)']['hits_3plus_trad'] / total) * 100
    p3_new = (stats['Modelo Nuevo (Solapamiento Ortogonal <= 3)']['hits_3plus_trad'] / total) * 100
    p3_rnd = (stats['Azar / Aleatorio (Línea Base)']['hits_3plus_trad'] / total) * 100
    print(f"{'% Sorteos con >= 3 Aciertos (Tradicional)':<42} | {p3_old:<15.1f}% | {p3_new:<15.1f}% | {p3_rnd:<.1f}%")

    p4_old = (stats['Modelo Anterior (Solapamiento Libre)']['hits_4plus_trad'] / total) * 100
    p4_new = (stats['Modelo Nuevo (Solapamiento Ortogonal <= 3)']['hits_4plus_trad'] / total) * 100
    p4_rnd = (stats['Azar / Aleatorio (Línea Base)']['hits_4plus_trad'] / total) * 100
    print(f"{'% Sorteos con >= 4 Aciertos (Tradicional)':<42} | {p4_old:<15.1f}% | {p4_new:<15.1f}% | {p4_rnd:<.1f}%")

    print("-" * 95)
    pe_old = np.mean(stats['Modelo Anterior (Solapamiento Libre)']['hits_pe'])
    pe_new = np.mean(stats['Modelo Nuevo (Solapamiento Ortogonal <= 3)']['hits_pe'])
    pe_rnd = np.mean(stats['Azar / Aleatorio (Línea Base)']['hits_pe'])
    print(f"{'Promedio Acierto Máx / Sorteo (Pozo Extra)':<42} | {pe_old:<16.2f} | {pe_new:<16.2f} | {pe_rnd:<.2f}")

    pe3_old = (stats['Modelo Anterior (Solapamiento Libre)']['hits_3plus_pe'] / total) * 100
    pe3_new = (stats['Modelo Nuevo (Solapamiento Ortogonal <= 3)']['hits_3plus_pe'] / total) * 100
    pe3_rnd = (stats['Azar / Aleatorio (Línea Base)']['hits_3plus_pe'] / total) * 100
    print(f"{'% Sorteos con >= 3 Aciertos (Pozo Extra)':<42} | {pe3_old:<15.1f}% | {pe3_new:<15.1f}% | {pe3_rnd:<.1f}%")

    pe4_old = (stats['Modelo Anterior (Solapamiento Libre)']['hits_4plus_pe'] / total) * 100
    pe4_new = (stats['Modelo Nuevo (Solapamiento Ortogonal <= 3)']['hits_4plus_pe'] / total) * 100
    pe4_rnd = (stats['Azar / Aleatorio (Línea Base)']['hits_4plus_pe'] / total) * 100
    print(f"{'% Sorteos con >= 4 Aciertos (Pozo Extra)':<42} | {pe4_old:<15.1f}% | {pe4_new:<15.1f}% | {pe4_rnd:<.1f}%")

    print("=" * 95)
    print(f" Tiempo total de simulación: {elapsed:.2f} segundos")
    print("=" * 95)

if __name__ == '__main__':
    run_walk_forward_backtest(n_eval_draws=100)
