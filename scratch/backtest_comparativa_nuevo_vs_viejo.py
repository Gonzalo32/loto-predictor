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
    compute_adaptive_pool
)

def run_backtest_head_to_head():
    csv_path = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
    draws = load_data(csv_path)
    if not draws:
        print("[ERROR] No se pudo leer historico_quini_completo.csv")
        return

    # Evaluaremos sobre los últimos 60 sorteos secuenciales reales (desde sorteo ~3358 hasta 3418)
    # 60 sorteos equivalen a más de 7 meses de juego bisemanal (Miércoles y Domingos).
    sorteos_disponibles = sorted(list(set(d['sorteo'] for d in draws)))
    eval_sorteos = sorteos_disponibles[-60:]

    print("="*95)
    print("      BACKTESTING COMPARATIVO HEAD-TO-HEAD: MODELO ANTERIOR VS MODELO NUEVO PREMIOS")
    print(f" Periodo de Evaluación: {len(eval_sorteos)} Sorteos Reales (Desde #{eval_sorteos[0]} hasta #{eval_sorteos[-1]})")
    print(" 3 Boletas por sorteo | Modalidades evaluadas: Tradicional, La Segunda, Revancha, Siempre Sale")
    print("="*95)

    stats_old = {
        'max_hits_modal': [],
        'hits_trad': [],
        'hits_seg': [],
        'hits_ss': [],
        'hits_pe': [],
        '3plus': 0,
        '4plus': 0,
        '5plus': 0,
        'premios_cobrados': 0,
        'detalle_premios': []
    }

    stats_new = {
        'max_hits_modal': [],
        'hits_trad': [],
        'hits_seg': [],
        'hits_ss': [],
        'hits_pe': [],
        '3plus': 0,
        '4plus': 0,
        '5plus': 0,
        'premios_cobrados': 0,
        'detalle_premios': []
    }

    start_time = time.time()

    for idx, target_sorteo in enumerate(eval_sorteos):
        # 1. Resultados reales del sorteo
        draw_group = [d for d in draws if d['sorteo'] == target_sorteo]
        mod_dict = {d['modalidad']: set(d['numbers']) for d in draw_group}
        target_trad = mod_dict.get('Tradicional', set())
        target_seg = mod_dict.get('Segunda', set())
        target_rev = mod_dict.get('Revancha', set())
        target_ss = mod_dict.get('SiempreSale', set())
        pozo_extra_set = target_trad.union(target_seg).union(target_rev)
        fecha_sorteo = draw_group[0]['fecha'] if draw_group else ''

        # 2. Datos de entrenamiento estrictamente anteriores
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        if len(train_draws) < 100:
            continue

        # 3. Features ML: Frecuencia 60, Recencia exponencial, Coocurrencia PCR, Momento de Retardo DDM
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1
                
        recency = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency[n] += weight

        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)

        # Delays y ciclo
        last_seen = {}
        for d_i, d in enumerate(train_draws):
            for n in d['numbers']: last_seen[n] = d_i
        T_now = len(train_draws)
        delays = {n: T_now - 1 - last_seen.get(n, 0) for n in range(46)}

        # Ranking Borda consolidado
        borda = defaultdict(float)
        models_to_rank = [
            (counts_60, 1.5),
            (recency, 2.0),
            (pcr_scores, 1.4),
            (ddm_scores, 1.2)
        ]
        for p_model, weight in models_to_rank:
            if isinstance(p_model, Counter):
                ranking_m = sorted(range(46), key=lambda x: p_model[x], reverse=True)
            else:
                ranking_m = sorted(range(46), key=lambda x: p_model[x], reverse=True)
            for rank, n in enumerate(ranking_m):
                borda[n] += weight * (45 - rank)

        borda_ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)

        # ==============================================================
        # ESTRATEGIA A: MODELO ANTERIOR (Pool 14 + Dispersión Forzada Overlap <= 3)
        # ==============================================================
        top_pool_old = sorted(borda_ranking[:14])
        combos_old = [c for c in itertools.combinations(top_pool_old, 6) if es_ticket_valido(c)]
        if not combos_old: combos_old = list(itertools.combinations(top_pool_old, 6))[:50]

        core_pool_old = set(borda_ranking[:8])
        surge_pool_old = set(top_pool_old) - core_pool_old

        v_core_old = [c for c in combos_old if sum(1 for n in c if n in core_pool_old) >= 4] or combos_old
        tA_old = max(v_core_old, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))

        v_B_old = [c for c in combos_old if len(set(c).intersection(set(tA_old))) <= 3] or combos_old
        tB_old = max(v_B_old, key=lambda b: sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))

        borda_C_old = borda.copy()
        for n in set(tA_old).union(set(tB_old)): borda_C_old[n] *= 0.5
        v_C_old = [c for c in combos_old if len(set(c).intersection(set(tA_old))) <= 3 and len(set(c).intersection(set(tB_old))) <= 3] or combos_old
        tC_old = max(v_C_old, key=lambda b: sum(borda_C_old[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool_old))

        tickets_old = [tA_old, tB_old, tC_old]

        # ==============================================================
        # ESTRATEGIA B: MODELO NUEVO (Pool Compacto 11 + Rueda Reducida Anclada)
        # ==============================================================
        # 1. Pool Adaptativo Compacto (10 a 12 números, media ~11)
        top_pool_new = compute_adaptive_pool(borda, alpha=0.22, min_pool=10, max_pool=12)
        top_pool_new_set = set(top_pool_new)

        combos_new = [c for c in itertools.combinations(sorted(top_pool_new), 6) if es_ticket_valido(c)]
        if not combos_new: combos_new = list(itertools.combinations(sorted(top_pool_new), 6))[:50]

        # 2. Puntuación de calidad intrínseca por combinación (Score Borda + Co-ocurrencia PCR)
        scored_combos_new = []
        for c in combos_new:
            s_nodes = sum(borda[n] for n in c)
            s_edges = sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
            scored_combos_new.append((s_nodes + 320.0 * s_edges, c))
        scored_combos_new.sort(key=lambda x: x[0], reverse=True)

        # Ticket A: Densidad máxima del Core
        tA_new = scored_combos_new[0][1]

        # Ticket B: Rueda anclada con solapamiento suave (3 a 4 números) para preservar el cuarteto ganador
        # pero rotando 2 números con alta afinidad mutua
        candidates_B = [c for _, c in scored_combos_new[1:] if 3 <= len(set(c).intersection(set(tA_new))) <= 4]
        if candidates_B:
            tB_new = candidates_B[0]
        else:
            candidates_B_alt = [c for _, c in scored_combos_new[1:] if 2 <= len(set(c).intersection(set(tA_new))) <= 4]
            tB_new = candidates_B_alt[0] if candidates_B_alt else scored_combos_new[1][1]

        # Ticket C: Optimizador de Cuartetos (Quad-Coverage Optimizer)
        # Maximiza cuartetos nuevos NO cubiertos por A y B dentro del pool de alta probabilidad
        quads_AB_new = set(itertools.combinations(tA_new, 4)).union(set(itertools.combinations(tB_new, 4)))
        
        candidates_C = []
        for s_val, c in scored_combos_new:
            if c == tA_new or c == tB_new:
                continue
            quads_c = set(itertools.combinations(c, 4))
            new_quads = len(quads_c - quads_AB_new)
            # Solapamiento balanceado (2 a 4 números con A y B)
            ovA = len(set(c).intersection(set(tA_new)))
            ovB = len(set(c).intersection(set(tB_new)))
            if 2 <= ovA <= 4 and 2 <= ovB <= 4:
                candidates_C.append((new_quads * 65.0 + s_val, c))

        if candidates_C:
            candidates_C.sort(key=lambda x: x[0], reverse=True)
            tC_new = candidates_C[0][1]
        else:
            candidates_C_fallback = [c for _, c in scored_combos_new if c != tA_new and c != tB_new]
            tC_new = candidates_C_fallback[0] if candidates_C_fallback else scored_combos_new[2][1]

        tickets_new = [tA_new, tB_new, tC_new]

        # ==============================================================
        # EVALUACIÓN DE ACERTOS Y PREMIOS OFICIALES QUINI 6
        # ==============================================================
        def evaluar_tickets(ticket_list):
            max_modal = 0
            max_tr = 0
            max_sg = 0
            max_ss = 0
            max_pe = 0
            premio = False
            premio_desc = []

            for t in ticket_list:
                t_set = set(t)
                h_tr = len(t_set.intersection(target_trad))
                h_sg = len(t_set.intersection(target_seg))
                h_rv = len(t_set.intersection(target_rev))
                h_ss = len(t_set.intersection(target_ss))
                h_pe = len(t_set.intersection(pozo_extra_set))

                max_tr = max(max_tr, h_tr)
                max_sg = max(max_sg, h_sg)
                max_ss = max(max_ss, h_ss)
                max_pe = max(max_pe, h_pe)
                max_modal = max(max_modal, h_tr, h_sg, h_rv, h_ss)

                # Condiciones Oficiales de Cobro de Premios en Efectivo:
                if h_tr >= 4:
                    premio = True
                    premio_desc.append(f"Tradicional {h_tr} aciertos")
                if h_sg >= 4:
                    premio = True
                    premio_desc.append(f"Segunda {h_sg} aciertos")
                if h_rv >= 6:
                    premio = True
                    premio_desc.append("Revancha POZO MAYOR 6 aciertos")
                if h_ss >= 5:
                    premio = True
                    premio_desc.append(f"Siempre Sale {h_ss} aciertos")
                if h_pe >= 6:
                    premio = True
                    premio_desc.append("Pozo Extra 6 aciertos")

            return max_modal, max_tr, max_sg, max_ss, max_pe, premio, premio_desc

        # Evaluar modelo viejo
        m_old, tr_old, sg_old, ss_old, pe_old, p_old, pdesc_old = evaluar_tickets(tickets_old)
        stats_old['max_hits_modal'].append(m_old)
        stats_old['hits_trad'].append(tr_old)
        stats_old['hits_seg'].append(sg_old)
        stats_old['hits_ss'].append(ss_old)
        stats_old['hits_pe'].append(pe_old)
        if m_old >= 3: stats_old['3plus'] += 1
        if m_old >= 4: stats_old['4plus'] += 1
        if m_old >= 5: stats_old['5plus'] += 1
        if p_old:
            stats_old['premios_cobrados'] += 1
            stats_old['detalle_premios'].append((target_sorteo, fecha_sorteo, pdesc_old))

        # Evaluar modelo nuevo
        m_new, tr_new, sg_new, ss_new, pe_new, p_new, pdesc_new = evaluar_tickets(tickets_new)
        stats_new['max_hits_modal'].append(m_new)
        stats_new['hits_trad'].append(tr_new)
        stats_new['hits_seg'].append(sg_new)
        stats_new['hits_ss'].append(ss_new)
        stats_new['hits_pe'].append(pe_new)
        if m_new >= 3: stats_new['3plus'] += 1
        if m_new >= 4: stats_new['4plus'] += 1
        if m_new >= 5: stats_new['5plus'] += 1
        if p_new:
            stats_new['premios_cobrados'] += 1
            stats_new['detalle_premios'].append((target_sorteo, fecha_sorteo, pdesc_new))

    elapsed = time.time() - start_time
    total = len(stats_old['max_hits_modal'])

    print("\n" + "="*95)
    print(f"               RESULTADOS FINALES DEL BACKTESTING (60 SORTEOS)")
    print(f" Tiempo de Simulación: {elapsed:.2f} s | Total Sorteos Analizados: {total}")
    print("="*95)

    print("\n1. COMPARATIVA GENERAL DE RENDIMIENTO POR BOLETAS (3 TICKETS)")
    print("-" * 95)
    print(f"{'Métrica':<40} | {'Modelo Anterior (Actual)':<23} | {'Modelo Nuevo Optimizado':<23}")
    print("-" * 95)
    avg_m_old = np.mean(stats_old['max_hits_modal'])
    avg_m_new = np.mean(stats_new['max_hits_modal'])
    print(f"{'Promedio Acierto Máximo Modal':<40} | {avg_m_old:<23.2f} | {avg_m_new:<23.2f}")

    p3_old = (stats_old['3plus'] / total) * 100
    p3_new = (stats_new['3plus'] / total) * 100
    print(f"{'Sorteos con >= 3 Aciertos':<40} | {stats_old['3plus']:>2} ({p3_old:4.1f}%)               | {stats_new['3plus']:>2} ({p3_new:4.1f}%)")

    p4_old = (stats_old['4plus'] / total) * 100
    p4_new = (stats_new['4plus'] / total) * 100
    print(f"{'Sorteos con >= 4 Aciertos (Premio Cash)':<40} | {stats_old['4plus']:>2} ({p4_old:4.1f}%)               | {stats_new['4plus']:>2} ({p4_new:4.1f}%)")

    p5_old = (stats_old['5plus'] / total) * 100
    p5_new = (stats_new['5plus'] / total) * 100
    print(f"{'Sorteos con >= 5 Aciertos':<40} | {stats_old['5plus']:>2} ({p5_old:4.1f}%)               | {stats_new['5plus']:>2} ({p5_new:4.1f}%)")

    prem_pct_old = (stats_old['premios_cobrados'] / total) * 100
    prem_pct_new = (stats_new['premios_cobrados'] / total) * 100
    print(f"{'Tasa Efectiva de Cobro de Premios':<40} | {stats_old['premios_cobrados']:>2} de {total} ({prem_pct_old:4.1f}%)        | {stats_new['premios_cobrados']:>2} de {total} ({prem_pct_new:4.1f}%)")
    print("-" * 95)

    print("\n2. DESGLOSE DE PROMEDIO DE ACIERTOS POR MODALIDAD")
    print("-" * 95)
    print(f"{'Modalidad':<40} | {'Modelo Anterior':<23} | {'Modelo Nuevo Optimizado':<23}")
    print("-" * 95)
    print(f"{'Tradicional Primer Sorteo':<40} | {np.mean(stats_old['hits_trad']):<23.2f} | {np.mean(stats_new['hits_trad']):<23.2f}")
    print(f"{'La Segunda del Quini':<40} | {np.mean(stats_old['hits_seg']):<23.2f} | {np.mean(stats_new['hits_seg']):<23.2f}")
    print(f"{'Siempre Sale':<40} | {np.mean(stats_old['hits_ss']):<23.2f} | {np.mean(stats_new['hits_ss']):<23.2f}")
    print(f"{'Pozo Extra (Unión 18 números)':<40} | {np.mean(stats_old['hits_pe']):<23.2f} | {np.mean(stats_new['hits_pe']):<23.2f}")
    print("-" * 95)

    print("\n3. PREMIOS REALES EN EFECTIVO COBRADOS:")
    print(" Modelo Anterior:")
    if stats_old['detalle_premios']:
        for s_num, s_f, descs in stats_old['detalle_premios']:
            print(f"   - Sorteo #{s_num} ({s_f}): {', '.join(descs)}")
    else:
        print("   - Ningún premio en efectivo cobrado (0 premios en 60 sorteos).")

    print("\n Modelo Nuevo Optimizado:")
    if stats_new['detalle_premios']:
        for s_num, s_f, descs in stats_new['detalle_premios']:
            print(f"   - Sorteo #{s_num} ({s_f}): {', '.join(descs)}")
    else:
        print("   - Ningún premio.")
    print("="*95)

    # Guardar en archivo scratch para consulta
    out_data = {
        'total_sorteos': total,
        'tiempo_seg': elapsed,
        'old': {
            'avg_max_hits': float(avg_m_old),
            'sorteos_3plus': stats_old['3plus'],
            'sorteos_4plus': stats_old['4plus'],
            'sorteos_5plus': stats_old['5plus'],
            'premios_cobrados': stats_old['premios_cobrados']
        },
        'new': {
            'avg_max_hits': float(avg_m_new),
            'sorteos_3plus': stats_new['3plus'],
            'sorteos_4plus': stats_new['4plus'],
            'sorteos_5plus': stats_new['5plus'],
            'premios_cobrados': stats_new['premios_cobrados']
        }
    }
    with open('scratch/resultado_backtest_head_to_head.json', 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)

if __name__ == '__main__':
    run_backtest_head_to_head()
