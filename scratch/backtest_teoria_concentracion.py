import csv
import json
import itertools
import numpy as np
from collections import defaultdict, Counter
import time
import os
import sys

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_attention_mlp_scores
)
from cobertura_combinatoria import select_optimal_wheel_triplet

def build_bibd_pool9_tickets(pool_9, borda, cooccur_matrix):
    """
    Construye 3 boletos usando un diseño balanceado 2-fold BIBD sobre Pool 9.
    Trío núcleo: pool_9[:3] = (A, B, C)
    Resto: pool_9[3:] = (D, E, F, G, H, I) (6 números)
    Cada boleto toma 2 números del trío núcleo y 4 números del resto.
    Cada número del pool aparece exactamente 2 veces en los 3 boletos.
    """
    core = list(pool_9[:3])
    rest = list(pool_9[3:9])
    
    # 3 pares de anclas:
    pairs = [
        (core[0], core[1]), # A, B
        (core[0], core[2]), # A, C
        (core[1], core[2])  # B, C
    ]
    
    # Repartir los 6 números del resto (D, E, F, G, H, I) en 3 subconjuntos de 4 números
    # de modo que cada número aparezca exactamente 2 veces.
    # Matemáticamente, el complemento de 4 en 6 es dejar 2 números afuera por boleto.
    # Dejamos afuera (D, E) en T1, (F, G) en T2, (H, I) en T3.
    # Encontrar la mejor partición del resto evaluando sinergia de coocurrencia:
    best_triplet = None
    best_score = -float('inf')
    
    # Evaluamos permutaciones de emparejamiento de los 6 del resto en 3 pares de exclusión
    # (D,E), (F,G), (H,I)
    pairs_of_rest = list(itertools.combinations(rest, 2))
    for p1 in pairs_of_rest:
        sub_rest1 = [n for n in rest if n not in p1] # 4 números
        t1 = tuple(sorted(list(pairs[0]) + sub_rest1))
        if not es_ticket_valido(t1):
            continue
            
        remaining_pairs = [p for p in pairs_of_rest if len(set(p) & set(p1)) == 0]
        for p2 in remaining_pairs:
            sub_rest2 = [n for n in rest if n not in p2]
            t2 = tuple(sorted(list(pairs[1]) + sub_rest2))
            if not es_ticket_valido(t2):
                continue
                
            p3 = tuple(sorted(list(set(rest) - set(p1) - set(p2))))
            sub_rest3 = [n for n in rest if n not in p3]
            t3 = tuple(sorted(list(pairs[2]) + sub_rest3))
            if not es_ticket_valido(t3):
                continue
                
            # Score conjunto de calidad: suma de Borda + coocurrencia
            total_s = 0.0
            for t in [t1, t2, t3]:
                total_s += sum(borda[n] for n in t)
                total_s += 350.0 * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(t, 2))
                
            if total_s > best_score:
                best_score = total_s
                best_triplet = [t1, t2, t3]
                
    if best_triplet is None:
        # Fallback voraz si restricciones de paridad/suma bloquearon partición exacta
        all_combos_9 = [c for c in itertools.combinations(pool_9, 6) if es_ticket_valido(c)]
        if not all_combos_9:
            all_combos_9 = list(itertools.combinations(pool_9, 6))[:50]
        all_combos_9.sort(key=lambda c: sum(borda[n] for n in c) + 300.0 * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(c, 2)), reverse=True)
        best_triplet = all_combos_9[:3]
        
    return best_triplet

def build_hierarchical_satellite_tickets(pool_10, borda, cooccur_matrix):
    """
    Construye 3 boletos jerárquicos:
    T1 = Mejor combinación pura del modelo.
    T2 y T3 anclan 4 números de T1 y rotan con las mejores alternativas del pool 10.
    """
    combos_10 = [c for c in itertools.combinations(pool_10, 6) if es_ticket_valido(c)]
    if not combos_10:
        combos_10 = list(itertools.combinations(pool_10, 6))[:50]
    combos_10.sort(key=lambda c: sum(borda[n] for n in c) + 350.0 * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(c, 2)), reverse=True)
    
    t1 = combos_10[0]
    t1_set = set(t1)
    
    # T2 mantiene 4 números de T1 con solapamiento 4
    cands_t2 = [c for c in combos_10[1:] if len(set(c) & t1_set) == 4]
    t2 = cands_t2[0] if cands_t2 else combos_10[1]
    
    # T3 mantiene 4 números de T1 (o 3 de T1 y 3 de alternativas)
    cands_t3 = [c for c in combos_10[1:] if c != t2 and len(set(c) & t1_set) == 4]
    t3 = cands_t3[0] if cands_t3 else (combos_10[2] if len(combos_10) > 2 else t2)
    
    return [t1, t2, t3]

def run_backtest_theory():
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]
    
    print("="*95)
    print(" BACKTEST WALK-FORWARD COMPARATIVO: VALIDACIÓN DE TEORÍA DE CONCENTRACIÓN")
    print(f" Evaluando {len(eval_draws)} sorteos históricos (# {eval_draws[0]['sorteo']} al #{eval_draws[-1]['sorteo']})")
    print(" Sin data leakage: cada sorteo se predice entrenando solo con el historial previo")
    print("="*95)
    
    strategies = {
        '1. Modelo Actual (Pool 12, Anti-SPOF, Overlap <= 3, Forzar 12 nums)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'pe_5+': 0,
            'max_hits_trad': [], 'pool_hits': [], 'desparramados': 0
        },
        '2. Propuesta: Pool 9 Compacto + Trío Núcleo BIBD 2-Fold': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'pe_5+': 0,
            'max_hits_trad': [], 'pool_hits': [], 'desparramados': 0
        },
        '3. Propuesta: Satélite Jerárquico (T1 Titular + Satélites Overlap 4)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'pe_5+': 0,
            'max_hits_trad': [], 'pool_hits': [], 'desparramados': 0
        },
        '4. Control Aleatorio (3 Boletos al Azar en Quini 6)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'pe_5+': 0,
            'max_hits_trad': [], 'pool_hits': [], 'desparramados': 0
        }
    }
    
    start_time = time.time()
    
    for step_i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        target_fecha = target_draw['fecha']
        
        # Sorteos reales en este sorteo
        group = [d for d in draws if d['sorteo'] == target_sorteo]
        real_by_mod = {d['modalidad']: set(d['numbers']) for d in group}
        target_trad = real_by_mod.get('Tradicional', set())
        target_seg = real_by_mod.get('Segunda', set())
        target_rev = real_by_mod.get('Revancha', set())
        target_ss = real_by_mod.get('SiempreSale', set())
        pozo_extra = target_trad | target_seg | target_rev
        
        # Historial de entrenamiento
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        
        # Generar puntuaciones de ensemble IA (Ventana 60 + Recencia Exp + Coocurrencia + Momentum + Atención)
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
            ({n: attn_scores[n] for n in range(46)}, 1.5)
        ]
        for m_scores, w in models_to_rank:
            ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
            for rank, n in enumerate(ranked):
                borda[n] += w * (45 - rank)
                
        ranking_full = sorted(range(46), key=lambda x: borda[x], reverse=True)
        
        # Pools
        pool_12 = sorted(ranking_full[:12])
        pool_10 = sorted(ranking_full[:10])
        pool_9  = sorted(ranking_full[:9], key=lambda x: borda[x], reverse=True) # ordenado por borda
        
        # 1. GENERAR BOLETOS ESTRATEGIA 1 (ACTUAL EN PRODUCCIÓN)
        combos_viable_12 = [c for c in itertools.combinations(pool_12, 6) if es_ticket_valido(c)]
        if not combos_viable_12:
            combos_viable_12 = list(itertools.combinations(pool_12, 6))[:50]
        tickets_actual = select_optimal_wheel_triplet(
            boletos_viables=combos_viable_12,
            pool=pool_12,
            scores=borda,
            cooccur_matrix=cooccur_exp_mat,
            max_overlap=3,
            w_cooccur=320.0
        )
        
        # 2. GENERAR BOLETOS ESTRATEGIA 2 (PROPUESTA BIBD POOL 9)
        tickets_bibd_9 = build_bibd_pool9_tickets(pool_9, borda, cooccur_exp_mat)
        
        # 3. GENERAR BOLETOS ESTRATEGIA 3 (SATÉLITE JERÁRQUICO POOL 10)
        tickets_satelite_10 = build_hierarchical_satellite_tickets(pool_10, borda, cooccur_exp_mat)
        
        # 4. GENERAR BOLETOS CONTROL ALEATORIO (Seed fija por sorteo para reproducibilidad)
        np.random.seed(target_sorteo)
        tickets_azar = []
        for _ in range(3):
            c_rand = tuple(sorted(np.random.choice(range(46), 6, replace=False)))
            tickets_azar.append(c_rand)
            
        strats_in_draw = [
            ('1. Modelo Actual (Pool 12, Anti-SPOF, Overlap <= 3, Forzar 12 nums)', tickets_actual, pool_12),
            ('2. Propuesta: Pool 9 Compacto + Trío Núcleo BIBD 2-Fold', tickets_bibd_9, pool_9),
            ('3. Propuesta: Satélite Jerárquico (T1 Titular + Satélites Overlap 4)', tickets_satelite_10, pool_10),
            ('4. Control Aleatorio (3 Boletos al Azar en Quini 6)', tickets_azar, [])
        ]
        
        for name, t_list, p_list in strats_in_draw:
            p_hits = len(set(p_list) & target_trad) if p_list else 0
            hits_per_ticket = [len(set(t) & target_trad) for t in t_list]
            max_h_trad = max(hits_per_ticket)
            
            # Evaluación en cualquiera de las 4 modalidades
            max_h_any = max(
                max(len(set(t) & m) for t in t_list)
                for m in [target_trad, target_seg, target_rev, target_ss] if m
            )
            
            # Evaluación en Pozo Extra
            max_h_pe = max(len(set(t) & pozo_extra) for t in t_list)
            
            # Chequear fenómeno "Desparramo": el pool metió 3 o más números, pero ningún boleto llegó a 3
            if p_hits >= 3 and max_h_trad < 3:
                strategies[name]['desparramados'] += 1
                
            strategies[name]['max_hits_trad'].append(max_h_trad)
            strategies[name]['pool_hits'].append(p_hits)
            if max_h_trad >= 3: strategies[name]['trad_3+'] += 1
            if max_h_trad >= 4: strategies[name]['trad_4+'] += 1
            if max_h_trad >= 5: strategies[name]['trad_5+'] += 1
            if max_h_any >= 4:  strategies[name]['any_4+'] += 1
            if max_h_pe >= 5:   strategies[name]['pe_5+'] += 1

    elapsed = time.time() - start_time
    total_s = len(eval_draws)
    
    print(f"\n[OK] Backtest finalizado en {elapsed:.1f} segundos.\n")
    print("="*105)
    print(f"{'Estrategia de Empaquetado':<54} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*105)
    for name, s in strategies.items():
        avg_h = np.mean(s['max_hits_trad'])
        p3 = (s['trad_3+'] / total_s) * 100
        n4 = s['trad_4+']
        p4 = (n4 / total_s) * 100
        n_any4 = s['any_4+']
        p_any4 = (n_any4 / total_s) * 100
        print(f"{name:<54} | {avg_h:<11.2f} | {p3:<9.1f}% | {n4} de {total_s} ({p4:4.1f}%)   | {n_any4} de {total_s} ({p_any4:4.1f}%)")
    print("="*105)
    
    print("\n" + "="*80)
    print(" ANÁLISIS DEL FENÓMENO DE DESPARRAMO (CUANDO EL POOL ACIERTA 3+ NÚMEROS):")
    print("="*80)
    for name in ['1. Modelo Actual (Pool 12, Anti-SPOF, Overlap <= 3, Forzar 12 nums)', 
                 '2. Propuesta: Pool 9 Compacto + Trío Núcleo BIBD 2-Fold',
                 '3. Propuesta: Satélite Jerárquico (T1 Titular + Satélites Overlap 4)']:
        desp = strategies[name]['desparramados']
        pe5 = strategies[name]['pe_5+']
        print(f" * {name[:50]}:")
        print(f"    - Sorteos donde los aciertos se desparramaron (Pool >= 3 pero Boletos < 3): {desp} sorteos")
        print(f"    - Sorteos con 5+ aciertos en Pozo Extra: {pe5} sorteos")
    print("="*80)

if __name__ == '__main__':
    run_backtest_theory()
