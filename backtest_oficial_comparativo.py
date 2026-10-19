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
    compute_attention_mlp_scores
)

def select_wheel_portfolio_optimo(pool_numbers, borda_scores, cooccur_mat, max_pool_size=12):
    """Implementa exactamente la lógica nueva de generador_cobertura_ml.py"""
    pool = sorted(pool_numbers[:max_pool_size])
    combos = [c for c in itertools.combinations(pool, 6) if es_ticket_valido(c)]
    if not combos:
        combos = [c for c in itertools.combinations(pool, 6)]
        
    scored_combos = []
    for c in combos:
        s_node = sum(borda_scores[n] for n in c)
        s_edge = sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
        scored_combos.append((s_node + 400.0 * s_edge, c))
    scored_combos.sort(key=lambda x: x[0], reverse=True)
    
    best_portfolio = None
    best_portfolio_score = -1e9
    candidates = scored_combos[:min(50, len(scored_combos))]
    
    for i in range(len(candidates)):
        c1 = candidates[i][1]
        s1 = candidates[i][0]
        set1 = set(c1)
        
        for j in range(i + 1, len(candidates)):
            c2 = candidates[j][1]
            set2 = set(c2)
            overlap_12 = len(set1.intersection(set2))
            if overlap_12 < 1 or overlap_12 > 3:
                continue
                
            inter_12 = set1.intersection(set2)
            s2 = candidates[j][0]
            union_12 = set1.union(set2)
            
            for k in range(j + 1, min(j + 30, len(candidates))):
                c3 = candidates[k][1]
                set3 = set(c3)
                
                # Regla Anti-SPOF: Ningún número en los 3 boletos a la vez
                if any(n in inter_12 for n in set3):
                    continue
                    
                overlap_13 = len(set1.intersection(set3))
                overlap_23 = len(set2.intersection(set3))
                if overlap_13 < 1 or overlap_13 > 3:
                    continue
                if overlap_23 < 1 or overlap_23 > 3:
                    continue
                    
                total_union = union_12.union(set3)
                coverage_bonus = len(total_union) * 250.0
                
                s3 = candidates[k][0]
                total_score = s1 + s2 + s3 + coverage_bonus
                
                if total_score > best_portfolio_score:
                    best_portfolio_score = total_score
                    best_portfolio = [c1, c2, c3]
                    
    if best_portfolio is None:
        best_portfolio = [scored_combos[0][1], scored_combos[1][1], scored_combos[2][1]]
    return best_portfolio

def run_official_backtest():
    csv_path = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
    draws = load_data(csv_path)
    if not draws:
        print("[ERROR] No se pudo leer historico_quini_completo.csv")
        return

    # Evaluaremos exactamente los últimos 50 sorteos de Quini 6
    trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
    eval_draws = trad_draws[-50:]
    
    print("="*90)
    print("      BACKTESTING COMPARATIVO OFICIAL: MODELO ANTERIOR VS MODELO NUEVO OPTIMIZADO")
    print(f" Evaluando {len(eval_draws)} sorteos históricos reales (Sorteo #{eval_draws[0]['sorteo']} al #{eval_draws[-1]['sorteo']})")
    print("="*90)

    # Métricas acumuladas:
    # 1. Modelo Anterior (Pool 14 + Core-Surge + Sin Anti-SPOF)
    # 2. Modelo Nuevo (Pool 12 + Wheel Portfolio + Anti-SPOF)
    # 3. Azar Puro (3 tickets aleatorios)
    # 4. Frecuencia Simple
    
    metrics = {
        'Modelo Anterior (Pool 14 Core-Surge)': {
            'hits_modal_max': [], 'hits_pe_max': [],
            '3plus_modal': 0, '4plus_modal': 0, '5plus_modal': 0,
            '3plus_pe': 0, '4plus_pe': 0, '5plus_pe': 0, '6plus_pe': 0
        },
        'Modelo Nuevo (Pool 14 Wheel Anti-SPOF)': {
            'hits_modal_max': [], 'hits_pe_max': [],
            '3plus_modal': 0, '4plus_modal': 0, '5plus_modal': 0,
            '3plus_pe': 0, '4plus_pe': 0, '5plus_pe': 0, '6plus_pe': 0
        },
        'Variante Pool 12 (Pool 12 Wheel Anti-SPOF)': {
            'hits_modal_max': [], 'hits_pe_max': [],
            '3plus_modal': 0, '4plus_modal': 0, '5plus_modal': 0,
            '3plus_pe': 0, '4plus_pe': 0, '5plus_pe': 0, '6plus_pe': 0
        },
        'Azar Puro (3 boletas aleatorias)': {
            'hits_modal_max': [], 'hits_pe_max': [],
            '3plus_modal': 0, '4plus_modal': 0, '5plus_modal': 0,
            '3plus_pe': 0, '4plus_pe': 0, '5plus_pe': 0, '6plus_pe': 0
        },
        'Frecuencia Simple (Top Frecuentes)': {
            'hits_modal_max': [], 'hits_pe_max': [],
            '3plus_modal': 0, '4plus_modal': 0, '5plus_modal': 0,
            '3plus_pe': 0, '4plus_pe': 0, '5plus_pe': 0, '6plus_pe': 0
        }
    }

    start_time = time.time()
    np.random.seed(42)

    for step_i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        
        # Resultados reales de las 4 modalidades y Pozo Extra para este sorteo
        sorteo_group = [d for d in draws if d['sorteo'] == target_sorteo]
        dict_mods = {d['modalidad']: set(d['numbers']) for d in sorteo_group}
        
        target_trad = dict_mods.get('Tradicional', set())
        target_seg = dict_mods.get('Segunda', set())
        target_rev = dict_mods.get('Revancha', set())
        target_ss = dict_mods.get('SiempreSale', set())
        pozo_extra_set = target_trad.union(target_seg).union(target_rev)

        # Historial de entrenamiento estrictamente anterior
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        if len(train_draws) < 100:
            continue

        # 1. Puntuaciones de Frecuencia y Recencia
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

        # Ranking Borda Consensuado
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

        # -----------------------------------------------------------------
        # ESTRATEGIA 1: MODELO ANTERIOR (Pool 14 + Core-Surge + Sin Anti-SPOF)
        # -----------------------------------------------------------------
        top14 = sorted(borda_ranking[:14])
        combos14 = [c for c in itertools.combinations(top14, 6) if es_ticket_valido(c)] or list(itertools.combinations(top14, 6))[:50]
        core_pool = set(borda_ranking[:8])
        surge_pool = set(top14) - core_pool
        
        v_core = [c for c in combos14 if sum(1 for n in c if n in core_pool) >= 4] or combos14
        tA_old = max(v_core, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))
        
        v_B = [c for c in combos14 if len(set(c).intersection(set(tA_old))) <= 3] or combos14
        tB_old = max(v_B, key=lambda b: sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))
        
        borda_C = borda.copy()
        for n in set(tA_old).union(set(tB_old)): borda_C[n] *= 0.5
        v_C = [c for c in combos14 if len(set(c).intersection(set(tA_old))) <= 3 and len(set(c).intersection(set(tB_old))) <= 3] or combos14
        tC_old = max(v_C, key=lambda b: sum(borda_C[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
        tickets_old = [tA_old, tB_old, tC_old]

        # -----------------------------------------------------------------
        # ESTRATEGIA 2: MODELO NUEVO (Pool 14 + Wheel Portfolio + Anti-SPOF)
        # -----------------------------------------------------------------
        tickets_new_14 = select_wheel_portfolio_optimo(borda_ranking, borda, cooccur_exp_mat, max_pool_size=14)

        # -----------------------------------------------------------------
        # ESTRATEGIA 3: VARIANTE POOL 12 (Pool 12 + Wheel Portfolio + Anti-SPOF)
        # -----------------------------------------------------------------
        tickets_new_12 = select_wheel_portfolio_optimo(borda_ranking, borda, cooccur_exp_mat, max_pool_size=12)

        # -----------------------------------------------------------------
        # ESTRATEGIA 4: AZAR PURO (3 boletas de 6 números)
        # -----------------------------------------------------------------
        tickets_rnd = [list(np.random.choice(46, 6, replace=False)) for _ in range(3)]

        # -----------------------------------------------------------------
        # ESTRATEGIA 5: FRECUENCIA SIMPLE (Top salidores históricos)
        # -----------------------------------------------------------------
        top_f = borda_ranking[:10]
        tickets_freq = [top_f[:6], top_f[1:7], top_f[2:8]]

        # EVALUACIÓN DE TODAS LAS ESTRATEGIAS
        all_strategies = [
            ('Modelo Anterior (Pool 14 Core-Surge)', tickets_old),
            ('Modelo Nuevo (Pool 14 Wheel Anti-SPOF)', tickets_new_14),
            ('Variante Pool 12 (Pool 12 Wheel Anti-SPOF)', tickets_new_12),
            ('Azar Puro (3 boletas aleatorias)', tickets_rnd),
            ('Frecuencia Simple (Top Frecuentes)', tickets_freq)
        ]

        for strat_name, t_list in all_strategies:
            # Máximos aciertos individuales en alguna de las 4 modalidades
            max_modal = 0
            # Máximos aciertos en Pozo Extra
            max_pe = 0
            
            for t in t_list:
                s = set(t)
                h_tr = len(s.intersection(target_trad))
                h_sg = len(s.intersection(target_seg))
                h_rv = len(s.intersection(target_rev))
                h_ss = len(s.intersection(target_ss))
                h_pe = len(s.intersection(pozo_extra_set))
                
                h_max_single = max(h_tr, h_sg, h_rv, h_ss)
                if h_max_single > max_modal: max_modal = h_max_single
                if h_pe > max_pe: max_pe = h_pe
                
            m = metrics[strat_name]
            m['hits_modal_max'].append(max_modal)
            m['hits_pe_max'].append(max_pe)
            
            if max_modal >= 3: m['3plus_modal'] += 1
            if max_modal >= 4: m['4plus_modal'] += 1
            if max_modal >= 5: m['5plus_modal'] += 1
            
            if max_pe >= 3: m['3plus_pe'] += 1
            if max_pe >= 4: m['4plus_pe'] += 1
            if max_pe >= 5: m['5plus_pe'] += 1
            if max_pe >= 6: m['6plus_pe'] += 1

        if (step_i + 1) % 10 == 0:
            print(f"Simulados {step_i + 1}/{len(eval_draws)} sorteos pasados...")

    elapsed = time.time() - start_time
    total = len(eval_draws)

    print("\n" + "="*95)
    print("               RESULTADOS DEL BACKTESTING COMPARATIVO OFICIAL (50 SORTEOS)")
    print("="*95)
    print(f" Tiempo Total de Ejecución Walk-Forward: {elapsed:.2f} segundos\n")

    print(f"{'Estrategia':<40} | {'Avg Aciertos':<12} | {'3+ Aciertos %':<14} | {'4+ Aciertos (Premios)':<22} | {'Pozo Extra 4+ %'}")
    print("-" * 105)

    for strat_name in metrics:
        m = metrics[strat_name]
        avg_h = np.mean(m['hits_modal_max'])
        p3 = (m['3plus_modal'] / total) * 100
        n4 = m['4plus_modal']
        p4 = (n4 / total) * 100
        p_pe4 = (m['4plus_pe'] / total) * 100
        print(f"{strat_name:<40} | {avg_h:<12.2f} | {p3:<5.1f}% ({m['3plus_modal']:2d}/{total}) | {n4:2d} ({p4:4.1f}%)               | {p_pe4:<5.1f}% ({m['4plus_pe']:2d}/{total})")
    print("="*105)

    # Actualizar resultado_backtest.json
    results_json = {
        'total_sorteos_evaluados': total,
        'tiempo_segundos': elapsed,
        'modelo_nuevo_pool14': {
            'promedio_aciertos_modal': float(np.mean(metrics['Modelo Nuevo (Pool 14 Wheel Anti-SPOF)']['hits_modal_max'])),
            'pct_3_plus_modal': float((metrics['Modelo Nuevo (Pool 14 Wheel Anti-SPOF)']['3plus_modal'] / total) * 100),
            'n_4_plus_modal': int(metrics['Modelo Nuevo (Pool 14 Wheel Anti-SPOF)']['4plus_modal']),
            'pct_4_plus_modal': float((metrics['Modelo Nuevo (Pool 14 Wheel Anti-SPOF)']['4plus_modal'] / total) * 100),
            'promedio_aciertos_pozo_extra': float(np.mean(metrics['Modelo Nuevo (Pool 14 Wheel Anti-SPOF)']['hits_pe_max'])),
            'pct_4_plus_pozo_extra': float((metrics['Modelo Nuevo (Pool 14 Wheel Anti-SPOF)']['4plus_pe'] / total) * 100)
        },
        'modelo_anterior': {
            'promedio_aciertos_modal': float(np.mean(metrics['Modelo Anterior (Pool 14 Core-Surge)']['hits_modal_max'])),
            'pct_3_plus_modal': float((metrics['Modelo Anterior (Pool 14 Core-Surge)']['3plus_modal'] / total) * 100),
            'n_4_plus_modal': int(metrics['Modelo Anterior (Pool 14 Core-Surge)']['4plus_modal']),
            'pct_4_plus_modal': float((metrics['Modelo Anterior (Pool 14 Core-Surge)']['4plus_modal'] / total) * 100),
            'promedio_aciertos_pozo_extra': float(np.mean(metrics['Modelo Anterior (Pool 14 Core-Surge)']['hits_pe_max'])),
            'pct_4_plus_pozo_extra': float((metrics['Modelo Anterior (Pool 14 Core-Surge)']['4plus_pe'] / total) * 100)
        },
        'azar': {
            'promedio_aciertos_modal': float(np.mean(metrics['Azar Puro (3 boletas aleatorias)']['hits_modal_max'])),
            'pct_3_plus_modal': float((metrics['Azar Puro (3 boletas aleatorias)']['3plus_modal'] / total) * 100),
            'n_4_plus_modal': int(metrics['Azar Puro (3 boletas aleatorias)']['4plus_modal'])
        }
    }

    with open('c:/Users/Administrador/Desktop/lot/resultado_backtest.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print("\n[OK] Métricas oficiales actualizadas en 'resultado_backtest.json'.")

if __name__ == '__main__':
    run_official_backtest()
