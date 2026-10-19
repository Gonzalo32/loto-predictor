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

def select_optimal_wheel_portfolio(pool_numbers, borda_scores, cooccur_mat, max_pool_size=11):
    """
    Selecciona un portafolio de 3 boletos con:
    1. Pool compacto (11 números máx) para evitar dilución combinatoria.
    2. Anti-SPOF: Ningún número puede aparecer en los 3 boletos (máx 2 repeticiones).
    3. Solapamiento balanceado: entre 1 y 3 números compartidos entre pares de boletos.
    4. Cobertura completa del pool: cada número del pool aparece al menos en 1 boleto.
    5. Agrupamiento por máxima afinidad de pares (coocurrencia histórica).
    """
    pool = sorted(pool_numbers[:max_pool_size])
    combos = [c for c in itertools.combinations(pool, 6) if es_ticket_valido(c)]
    if not combos:
        combos = [c for c in itertools.combinations(pool, 6)]
        
    # Calificación intrínseca de cada boleto viable
    scored = []
    for c in combos:
        s_node = sum(borda_scores[n] for n in c)
        s_edge = sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
        scored.append((s_node + 400.0 * s_edge, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Búsqueda combinatoria global de la mejor terna (T1, T2, T3)
    best_portfolio = None
    best_portfolio_score = -1e9
    
    # Consideramos los top 40 boletos viables candidatos para optimizar la terna
    candidates = scored[:45]
    
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
            
            for k in range(j + 1, min(j + 25, len(candidates))):
                c3 = candidates[k][1]
                set3 = set(c3)
                
                # Regla Anti-SPOF: Ningún número puede estar en los 3 tickets simultáneamente
                if any(n in inter_12 for n in set3):
                    continue
                    
                overlap_13 = len(set1.intersection(set3))
                overlap_23 = len(set2.intersection(set3))
                
                if overlap_13 < 1 or overlap_13 > 3:
                    continue
                if overlap_23 < 1 or overlap_23 > 3:
                    continue
                    
                total_union = union_12.union(set3)
                # Bonificación por cubrir todo el pool (11 números)
                coverage_bonus = len(total_union) * 250.0
                
                s3 = candidates[k][0]
                total_score = s1 + s2 + s3 + coverage_bonus
                
                if total_score > best_portfolio_score:
                    best_portfolio_score = total_score
                    best_portfolio = [c1, c2, c3]
                    
    if best_portfolio is None:
        # Fallback seguro
        best_portfolio = [scored[0][1], scored[1][1], scored[2][1]]
        
    return best_portfolio

def run_test():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    print("="*80)
    print(" EVALUACIÓN DEL NUEVO OPTIMIZADOR EN EL HISTORIAL COMPLETO (ÚLTIMOS 45 SORTEOS)")
    print("="*80)
    
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-45:]
    
    stats_old = {'3plus': 0, '4plus': 0, '5plus': 0, 'max_hits': []}
    stats_new = {'3plus': 0, '4plus': 0, '5plus': 0, 'max_hits': []}
    
    for count, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        group = [d for d in draws if d['sorteo'] == target_sorteo]
        real_mods = {d['modalidad']: set(d['numbers']) for d in group}
        target_trad = real_mods.get('Tradicional', set())
        target_seg = real_mods.get('Segunda', set())
        target_rev = real_mods.get('Revancha', set())
        target_ss = real_mods.get('SiempreSale', set())
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        if len(train_draws) < 100:
            continue
            
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
        
        # 1. GENERADOR ANTERIOR (Pool 14 + Core-Surge)
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
        
        # 2. GENERADOR NUEVO (Pool 11 Compacto + Anti-SPOF + Wheel Portfolio Global)
        tickets_new = select_optimal_wheel_portfolio(borda_ranking, borda, cooccur_exp_mat, max_pool_size=11)
        
        # Evaluar
        for t_list, st in [(tickets_old, stats_old), (tickets_new, stats_new)]:
            max_hit = 0
            for t in t_list:
                s = set(t)
                h = max(
                    len(s.intersection(target_trad)),
                    len(s.intersection(target_seg)),
                    len(s.intersection(target_rev)),
                    len(s.intersection(target_ss))
                )
                if h > max_hit: max_hit = h
            st['max_hits'].append(max_hit)
            if max_hit >= 3: st['3plus'] += 1
            if max_hit >= 4: st['4plus'] += 1
            if max_hit >= 5: st['5plus'] += 1
            
        if (count + 1) % 15 == 0:
            print(f"Evaluados {count + 1}/{len(eval_draws)} sorteos...")
            
    total = len(eval_draws)
    print("\n" + "="*80)
    print("RESULTADOS FINALES DE LA COMPARACIÓN:")
    print("="*80)
    print(f"1. GENERADOR ANTERIOR (Pool 14 + Core-Surge):")
    print(f"   - Promedio Máximo Acierto: {np.mean(stats_old['max_hits']):.2f}")
    print(f"   - Tasa 3+ Aciertos: {stats_old['3plus']}/{total} ({(stats_old['3plus']/total)*100:.1f}%)")
    print(f"   - Tasa 4+ Aciertos (Premios): {stats_old['4plus']}/{total} ({(stats_old['4plus']/total)*100:.1f}%)")
    print(f"   - Tasa 5+ Aciertos: {stats_old['5plus']}/{total} ({(stats_old['5plus']/total)*100:.1f}%)")
    
    print(f"\n2. GENERADOR NUEVO (Pool 11 Compacto + Anti-SPOF + Wheel Portfolio):")
    print(f"   - Promedio Máximo Acierto: {np.mean(stats_new['max_hits']):.2f}")
    print(f"   - Tasa 3+ Aciertos: {stats_new['3plus']}/{total} ({(stats_new['3plus']/total)*100:.1f}%)")
    print(f"   - Tasa 4+ Aciertos (Premios): {stats_new['4plus']}/{total} ({(stats_new['4plus']/total)*100:.1f}%)")
    print(f"   - Tasa 5+ Aciertos: {stats_new['5plus']}/{total} ({(stats_new['5plus']/total)*100:.1f}%)")
    print("="*80)

if __name__ == '__main__':
    run_test()
