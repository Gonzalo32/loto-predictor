import csv
import itertools
import numpy as np
from collections import defaultdict, Counter
import time
import sys

sys.path.insert(0, 'c:/Users/Administrador/Desktop/lot')
from generador_cobertura_ml import (
    load_data, es_ticket_valido,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_adaptive_pool
)

def run_trimodal_backtest():
    csv_path = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
    draws = load_data(csv_path)
    sorteos_disponibles = sorted(list(set(d['sorteo'] for d in draws)))
    eval_sorteos = sorteos_disponibles[-80:]

    print("="*95)
    print("BACKTEST: MODELO ACTUAL VS MODELO ESPECIALIZADO TRI-MODAL (80 SORTEOS)")
    print("="*95)

    stats = {
        'Modelo Anterior (Genérico Disperso)': {'3plus': 0, '4plus': 0, 'premios': 0, 'max_hits': []},
        'Modelo Nuevo A (Rueda Reducida Anclada)': {'3plus': 0, '4plus': 0, 'premios': 0, 'max_hits': []},
        'Modelo Nuevo B (Especializado Tri-Modal: T1 Trad, T2 Seg, T3 SS)': {'3plus': 0, '4plus': 0, 'premios': 0, 'max_hits': []},
    }

    for target_sorteo in eval_sorteos:
        draw_group = [d for d in draws if d['sorteo'] == target_sorteo]
        mod_dict = {d['modalidad']: set(d['numbers']) for d in draw_group}
        target_trad = mod_dict.get('Tradicional', set())
        target_seg = mod_dict.get('Segunda', set())
        target_rev = mod_dict.get('Revancha', set())
        target_ss = mod_dict.get('SiempreSale', set())
        pozo_extra_set = target_trad.union(target_seg).union(target_rev)

        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        if len(train_draws) < 100:
            continue

        # Global Features
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1
                
        recency = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency[n] += weight

        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)

        borda = defaultdict(float)
        for p_model, w in [(counts_60, 1.5), (recency, 2.0), (pcr_scores, 1.4), (ddm_scores, 1.2)]:
            rk = sorted(range(46), key=lambda x: p_model[x], reverse=True)
            for rank, n in enumerate(rk): borda[n] += w * (45 - rank)

        borda_ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)

        # -------------------------------------------------------------
        # 1. MODELO ANTERIOR (Pool 14 + overlap <= 3)
        # -------------------------------------------------------------
        top_pool_old = sorted(borda_ranking[:14])
        combos_old = [c for c in itertools.combinations(top_pool_old, 6) if es_ticket_valido(c)] or list(itertools.combinations(top_pool_old, 6))[:50]
        core_old = set(borda_ranking[:8])
        surge_old = set(top_pool_old) - core_old

        v_core_old = [c for c in combos_old if sum(1 for n in c if n in core_old) >= 4] or combos_old
        tA_1 = max(v_core_old, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
        v_B_old = [c for c in combos_old if len(set(c).intersection(set(tA_1))) <= 3] or combos_old
        tB_1 = max(v_B_old, key=lambda b: sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))
        borda_C_1 = borda.copy()
        for n in set(tA_1).union(set(tB_1)): borda_C_1[n] *= 0.5
        v_C_old = [c for c in combos_old if len(set(c).intersection(set(tA_1))) <= 3 and len(set(c).intersection(set(tB_1))) <= 3] or combos_old
        tC_1 = max(v_C_old, key=lambda b: sum(borda_C_1[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_old))
        tickets_1 = [tA_1, tB_1, tC_1]

        # -------------------------------------------------------------
        # 2. MODELO NUEVO A (Rueda Reducida Anclada en Pool 11)
        # -------------------------------------------------------------
        top_pool_new = compute_adaptive_pool(borda, alpha=0.20, min_pool=10, max_pool=12)
        combos_new = [c for c in itertools.combinations(sorted(top_pool_new), 6) if es_ticket_valido(c)] or list(itertools.combinations(sorted(top_pool_new), 6))[:50]

        scored_new = []
        for c in combos_new:
            s_nodes = sum(borda[n] for n in c)
            s_edges = sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2))
            scored_new.append((s_nodes + 320.0 * s_edges, c))
        scored_new.sort(key=lambda x: x[0], reverse=True)

        tA_2 = scored_new[0][1]
        cand_B = [c for _, c in scored_new[1:] if 3 <= len(set(c).intersection(set(tA_2))) <= 4]
        tB_2 = cand_B[0] if cand_B else scored_new[1][1]

        quads_AB_2 = set(itertools.combinations(tA_2, 4)).union(set(itertools.combinations(tB_2, 4)))
        cand_C = []
        for s_val, c in scored_new:
            if c == tA_2 or c == tB_2: continue
            new_quads = len(set(itertools.combinations(c, 4)) - quads_AB_2)
            cand_C.append((new_quads * 75.0 + s_val, c))
        cand_C.sort(key=lambda x: x[0], reverse=True)
        tC_2 = cand_C[0][1] if cand_C else scored_new[2][1]
        tickets_2 = [tA_2, tB_2, tC_2]

        # -------------------------------------------------------------
        # 3. MODELO NUEVO B (Especializado Tri-Modal)
        # T1: Diseñado para TRADICIONAL
        # T2: Diseñado para LA SEGUNDA
        # T3: Diseñado para SIEMPRE SALE / CONSENSO GLOBAL
        # -------------------------------------------------------------
        trad_counts = Counter()
        seg_counts = Counter()
        ss_counts = Counter()
        for d in train_draws[-50:]:
            if d['modalidad'] == 'Tradicional':
                for n in d['numbers']: trad_counts[n] += 1
            elif d['modalidad'] == 'Segunda':
                for n in d['numbers']: seg_counts[n] += 1
            elif d['modalidad'] == 'SiempreSale':
                for n in d['numbers']: ss_counts[n] += 1

        # T1 Tradicional: sesgo de frecuencia y pares de Tradicional
        t1_3 = max(combos_new, key=lambda c: sum(borda[n] + 16.0 * trad_counts[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        
        # T2 Segunda: sesgo de frecuencia y pares de Segunda (manteniendo overlap 3-4 para anclaje)
        t2_3 = max(combos_new, key=lambda c: sum(borda[n] + 16.0 * seg_counts[n] for n in c) + 380.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        
        # T3 Siempre Sale / Consenso
        v_ss = [c for c in combos_new if c != t1_3 and c != t2_3] or combos_new
        t3_3 = max(v_ss, key=lambda c: sum(borda[n] + 12.0 * ss_counts[n] for n in c) + 300.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        tickets_3 = [t1_3, t2_3, t3_3]

        all_engines = [
            ('Modelo Anterior (Genérico Disperso)', tickets_1),
            ('Modelo Nuevo A (Rueda Reducida Anclada)', tickets_2),
            ('Modelo Nuevo B (Especializado Tri-Modal: T1 Trad, T2 Seg, T3 SS)', tickets_3)
        ]

        for name, t_list in all_engines:
            max_m = 0
            premio = False
            for t in t_list:
                t_set = set(t)
                h_tr = len(t_set.intersection(target_trad))
                h_sg = len(t_set.intersection(target_seg))
                h_rv = len(t_set.intersection(target_rev))
                h_ss = len(t_set.intersection(target_ss))
                h_pe = len(t_set.intersection(pozo_extra_set))

                max_m = max(max_m, h_tr, h_sg, h_rv, h_ss)
                if h_tr >= 4 or h_sg >= 4 or h_rv >= 6 or h_ss >= 5 or h_pe >= 6:
                    premio = True

            stats[name]['max_hits'].append(max_m)
            if max_m >= 3: stats[name]['3plus'] += 1
            if max_m >= 4: stats[name]['4plus'] += 1
            if premio: stats[name]['premios'] += 1

    total = len(eval_sorteos)
    print("\n" + "="*105)
    print(f"{'Estrategia de Empaquetado':<56} | {'Avg Max Hit':<12} | {'3+ Hits %':<12} | {'4+ Hits':<10} | {'Sorteos Premiados'}")
    print("="*105)
    for name in stats:
        s = stats[name]
        avg_h = np.mean(s['max_hits'])
        p3 = (s['3plus'] / total) * 100
        p4 = (s['4plus'] / total) * 100
        prem_p = (s['premios'] / total) * 100
        print(f"{name:<56} | {avg_h:<12.2f} | {s['3plus']:>2} ({p3:4.1f}%)     | {s['4plus']:>2} ({p4:4.1f}%)    | {s['premios']}/{total} ({prem_p:4.1f}%)")
    print("="*105)

if __name__ == '__main__':
    run_trimodal_backtest()
