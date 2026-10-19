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

def run_large_backtest():
    csv_path = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
    draws = load_data(csv_path)
    if not draws:
        print("[ERROR] No se pudo leer historico_quini_completo.csv")
        return

    # Evaluaremos sobre los últimos 80 sorteos secuenciales reales (~10 meses de juego real)
    sorteos_disponibles = sorted(list(set(d['sorteo'] for d in draws)))
    eval_sorteos = sorteos_disponibles[-80:]

    print("="*95)
    print("      BACKTESTING MASIVO REAL: MODELO ANTERIOR VS MODELO NUEVO (80 SORTEOS)")
    print(f" Periodo: {len(eval_sorteos)} Sorteos Reales (Desde #{eval_sorteos[0]} hasta #{eval_sorteos[-1]})")
    print(" Evaluando las 4 modalidades: Tradicional, La Segunda, Revancha, Siempre Sale")
    print("="*95)

    stats_old = {
        'max_hits': [],
        'hits_trad': [], 'hits_seg': [], 'hits_ss': [], 'hits_pe': [],
        '3plus': 0, '4plus': 0, '5plus': 0,
        'premios_cobrados': 0,
        'premios_detalle': []
    }

    stats_new = {
        'max_hits': [],
        'hits_trad': [], 'hits_seg': [], 'hits_ss': [], 'hits_pe': [],
        '3plus': 0, '4plus': 0, '5plus': 0,
        'premios_cobrados': 0,
        'premios_detalle': []
    }

    start_time = time.time()

    for idx, target_sorteo in enumerate(eval_sorteos):
        draw_group = [d for d in draws if d['sorteo'] == target_sorteo]
        mod_dict = {d['modalidad']: set(d['numbers']) for d in draw_group}
        target_trad = mod_dict.get('Tradicional', set())
        target_seg = mod_dict.get('Segunda', set())
        target_rev = mod_dict.get('Revancha', set())
        target_ss = mod_dict.get('SiempreSale', set())
        pozo_extra_set = target_trad.union(target_seg).union(target_rev)
        fecha_sorteo = draw_group[0]['fecha'] if draw_group else ''

        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        if len(train_draws) < 100:
            continue

        # Features del Modelo
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1
                
        recency = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency[n] += weight

        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)

        # Borda Score Combinado
        borda = defaultdict(float)
        models_to_rank = [
            (counts_60, 1.5),
            (recency, 2.0),
            (pcr_scores, 1.4),
            (ddm_scores, 1.2)
        ]
        for p_model, weight in models_to_rank:
            ranking_m = sorted(range(46), key=lambda x: p_model[x], reverse=True)
            for rank, n in enumerate(ranking_m):
                borda[n] += weight * (45 - rank)

        borda_ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)

        # ------------------------------------------------------------------
        # MODELO ANTERIOR (Pool 14 + Dispersión Forzada Overlap <= 3 + Borda_C * 0.5)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # MODELO NUEVO (Pool 11 Adaptativo + Rueda Reducida Anclada de Cuartetos)
        # ------------------------------------------------------------------
        top_pool_new = compute_adaptive_pool(borda, alpha=0.20, min_pool=10, max_pool=12)
        combos_new = [c for c in itertools.combinations(sorted(top_pool_new), 6) if es_ticket_valido(c)]
        if not combos_new: combos_new = list(itertools.combinations(sorted(top_pool_new), 6))[:50]

        scored_new = []
        for c in combos_new:
            s_nodes = sum(borda[n] for n in c)
            s_edges = sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
            scored_new.append((s_nodes + 320.0 * s_edges, c))
        scored_new.sort(key=lambda x: x[0], reverse=True)

        # Ticket A: Densidad Core Máxima
        tA_new = scored_new[0][1]

        # Ticket B: Rueda con solapamiento 3 a 4 con A (retiene el cuarteto ganador)
        cand_B = [c for _, c in scored_new[1:] if 3 <= len(set(c).intersection(set(tA_new))) <= 4]
        if cand_B:
            tB_new = cand_B[0]
        else:
            cand_B_fallback = [c for _, c in scored_new[1:] if 2 <= len(set(c).intersection(set(tA_new))) <= 4]
            tB_new = cand_B_fallback[0] if cand_B_fallback else scored_new[1][1]

        # Ticket C: Optimizador de Cuartetos sin destruir scores
        quads_AB = set(itertools.combinations(tA_new, 4)).union(set(itertools.combinations(tB_new, 4)))
        cand_C = []
        for s_val, c in scored_new:
            if c == tA_new or c == tB_new: continue
            quads_c = set(itertools.combinations(c, 4))
            new_quads = len(quads_c - quads_AB)
            ovA = len(set(c).intersection(set(tA_new)))
            ovB = len(set(c).intersection(set(tB_new)))
            if 2 <= ovA <= 4 and 2 <= ovB <= 4:
                cand_C.append((new_quads * 75.0 + s_val, c))

        if cand_C:
            cand_C.sort(key=lambda x: x[0], reverse=True)
            tC_new = cand_C[0][1]
        else:
            cand_C_fallback = [c for _, c in scored_new if c != tA_new and c != tB_new]
            tC_new = cand_C_fallback[0] if cand_C_fallback else scored_new[2][1]

        tickets_new = [tA_new, tB_new, tC_new]

        # ------------------------------------------------------------------
        # EVALUACIÓN DE AMBOS
        # ------------------------------------------------------------------
        def eval_set(t_list):
            max_m = 0
            m_tr = max(len(set(t).intersection(target_trad)) for t in t_list)
            m_sg = max(len(set(t).intersection(target_seg)) for t in t_list)
            m_rv = max(len(set(t).intersection(target_rev)) for t in t_list)
            m_ss = max(len(set(t).intersection(target_ss)) for t in t_list)
            m_pe = max(len(set(t).intersection(pozo_extra_set)) for t in t_list)
            max_m = max(m_tr, m_sg, m_rv, m_ss)

            premio = False
            p_desc = []
            if m_tr >= 4: premio = True; p_desc.append(f"Tradicional {m_tr} aciertos")
            if m_sg >= 4: premio = True; p_desc.append(f"Segunda {m_sg} aciertos")
            if m_rv >= 6: premio = True; p_desc.append("Revancha 6 aciertos (Jackpot)")
            if m_ss >= 5: premio = True; p_desc.append(f"Siempre Sale {m_ss} aciertos")
            if m_pe >= 6: premio = True; p_desc.append("Pozo Extra 6 aciertos")

            return max_m, m_tr, m_sg, m_ss, m_pe, premio, p_desc

        m_o, tr_o, sg_o, ss_o, pe_o, p_o, pdesc_o = eval_set(tickets_old)
        stats_old['max_hits'].append(m_o)
        stats_old['hits_trad'].append(tr_o)
        stats_old['hits_seg'].append(sg_o)
        stats_old['hits_ss'].append(ss_o)
        stats_old['hits_pe'].append(pe_o)
        if m_o >= 3: stats_old['3plus'] += 1
        if m_o >= 4: stats_old['4plus'] += 1
        if m_o >= 5: stats_old['5plus'] += 1
        if p_o:
            stats_old['premios_cobrados'] += 1
            stats_old['premios_detalle'].append((target_sorteo, fecha_sorteo, pdesc_o))

        m_n, tr_n, sg_n, ss_n, pe_n, p_n, pdesc_n = eval_set(tickets_new)
        stats_new['max_hits'].append(m_n)
        stats_new['hits_trad'].append(tr_n)
        stats_new['hits_seg'].append(sg_n)
        stats_new['hits_ss'].append(ss_n)
        stats_new['hits_pe'].append(pe_n)
        if m_n >= 3: stats_new['3plus'] += 1
        if m_n >= 4: stats_new['4plus'] += 1
        if m_n >= 5: stats_new['5plus'] += 1
        if p_n:
            stats_new['premios_cobrados'] += 1
            stats_new['premios_detalle'].append((target_sorteo, fecha_sorteo, pdesc_n))

    elapsed = time.time() - start_time
    total = len(stats_old['max_hits'])

    print(f"\nSimulación completada en {elapsed:.2f} segundos.")
    print("="*95)
    print(f"{'Métrica de Desempeño (80 Sorteos)':<42} | {'Modelo Anterior':<22} | {'Modelo Nuevo':<22} | {'Mejora'}")
    print("="*95)
    
    avg_old = np.mean(stats_old['max_hits'])
    avg_new = np.mean(stats_new['max_hits'])
    print(f"{'Promedio Acierto Máximo por Sorteo':<42} | {avg_old:<22.2f} | {avg_new:<22.2f} | {((avg_new-avg_old)/avg_old)*100:+.1f}%")

    pct3_old = (stats_old['3plus'] / total) * 100
    pct3_new = (stats_new['3plus'] / total) * 100
    print(f"{'Sorteos con >= 3 Aciertos':<42} | {stats_old['3plus']:>2} ({pct3_old:4.1f}%)              | {stats_new['3plus']:>2} ({pct3_new:4.1f}%)              | {pct3_new - pct3_old:+.1f}%")

    pct4_old = (stats_old['4plus'] / total) * 100
    pct4_new = (stats_new['4plus'] / total) * 100
    print(f"{'Sorteos con >= 4 Aciertos (Premio Cash)':<42} | {stats_old['4plus']:>2} ({pct4_old:4.1f}%)              | {stats_new['4plus']:>2} ({pct4_new:4.1f}%)              | {pct4_new - pct4_old:+.1f}%")

    pct_prem_old = (stats_old['premios_cobrados'] / total) * 100
    pct_prem_new = (stats_new['premios_cobrados'] / total) * 100
    print(f"{'Efectividad de Cobro de Premios':<42} | {stats_old['premios_cobrados']:>2} de {total} ({pct_prem_old:4.1f}%)       | {stats_new['premios_cobrados']:>2} de {total} ({pct_prem_new:4.1f}%)       | {pct_prem_new - pct_prem_old:+.1f}%")
    print("="*95)

    print("\n[DETALLE DE PREMIOS EN EFECTIVO COBRADOS]")
    print(f"Modelo Anterior ({stats_old['premios_cobrados']} premios):")
    for s_num, s_f, descs in stats_old['premios_detalle']:
        print(f"  - Sorteo #{s_num} ({s_f}): {', '.join(descs)}")
    print(f"\nModelo Nuevo ({stats_new['premios_cobrados']} premios):")
    for s_num, s_f, descs in stats_new['premios_detalle']:
        print(f"  - Sorteo #{s_num} ({s_f}): {', '.join(descs)}")
    print("="*95)

if __name__ == '__main__':
    run_large_backtest()
