import csv
import itertools
import numpy as np
from collections import defaultdict, Counter

def load_data(filepath='historico_quini_completo.csv'):
    draws = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                draws.append({
                    'sorteo': int(row['Sorteo']),
                    'fecha': row['Fecha'],
                    'modalidad': row['Modalidad'].strip(),
                    'numbers': sorted([int(row[f'B{i}']) for i in range(1, 7)])
                })
            except (ValueError, KeyError):
                continue
    modality_order = {'Tradicional': 1, 'Segunda': 2, 'Revancha': 3, 'SiempreSale': 4}
    draws.sort(key=lambda x: (x['sorteo'], modality_order.get(x['modalidad'], 9)))
    return draws

def es_ticket_valido(ticket):
    ticket = sorted(ticket)
    # 1. Consecutividad: max 2 parejas consecutivas
    gaps_le_1 = sum(1 for i in range(1, len(ticket)) if ticket[i] - ticket[i-1] <= 1)
    if gaps_le_1 > 2:
        return False
    # 2. Spread: 15..45
    spread = ticket[5] - ticket[0]
    if spread < 15 or spread > 45:
        return False
    # 3. Decenas: max 3 números por decena
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(count > 3 for count in decenas.values()):
        return False
    # 4. Paridad: entre 1 y 5 pares
    pares = sum(1 for num in ticket if num % 2 == 0)
    if pares < 1 or pares > 5:
        return False
    # 5. Suma: entre 80 y 190
    s = sum(ticket)
    if s < 80 or s > 190:
        return False
    return True

draws = load_data()
sorteos_unicos = sorted(list(set(d['sorteo'] for d in draws)))
eval_sorteos = sorteos_unicos[-60:]

results = {
    '1. Cobertura Total (Actual: Anti-SPOF, Overlap <= 3, Forzar 12 nums)': {'3+': 0, '4+': 0, '5+': 0, 'premios': 0, 'distribucion_hits': []},
    '2. Satélite Jerárquico (T1 = Titular, T2 y T3 anclan 4 del T1)': {'3+': 0, '4+': 0, '5+': 0, 'premios': 0, 'distribucion_hits': []},
    '3. Rueda con 2 Anclas Fijas (Top 2 en los 3 tickets + rotación Pool)': {'3+': 0, '4+': 0, '5+': 0, 'premios': 0, 'distribucion_hits': []},
    '4. Rueda con 3 Anclas Fijas (Top 3 en los 3 tickets + rotación Pool)': {'3+': 0, '4+': 0, '5+': 0, 'premios': 0, 'distribucion_hits': []},
}

for s_num in eval_sorteos:
    train_draws = [d for d in draws if d['sorteo'] < s_num]
    real_draws = [d for d in draws if d['sorteo'] == s_num]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in real_draws}
    target_real = real_by_mod.get('Tradicional', set())
    if not target_real:
        continue

    # Scores
    counts_60 = Counter()
    for d in train_draws[-60:]:
        for n in d['numbers']: counts_60[n] += 1
    recency = np.zeros(46)
    for t_idx, d in enumerate(train_draws[-30:]):
        w = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency[n] += w
    cooccur = defaultdict(lambda: defaultdict(float))
    for t_idx, d in enumerate(train_draws[-60:]):
        w = np.exp((t_idx - 60) / 15.0)
        for u, v in itertools.combinations(d['numbers'], 2):
            cooccur[u][v] += w
            cooccur[v][u] += w

    borda = {}
    for n in range(46):
        borda[n] = counts_60[n] * 1.5 + recency[n] * 2.5
    ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)
    pool_12 = sorted(ranking[:12])

    combos_12 = [c for c in itertools.combinations(pool_12, 6) if es_ticket_valido(c)]
    if not combos_12: combos_12 = list(itertools.combinations(pool_12, 6))[:50]
    combos_12.sort(key=lambda c: sum(borda[n] for n in c) + 250.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)), reverse=True)

    # --- Estrategia 1: Actual Cobertura Total ---
    t1_1 = combos_12[0]
    t2_1 = next((c for c in combos_12 if len(set(c) & set(t1_1)) <= 3), combos_12[1])
    uncovered = set(pool_12) - (set(t1_1) | set(t2_1))
    t3_1 = next((c for c in combos_12 if uncovered.issubset(set(c)) and len(set(c) & set(t1_1)) <= 3 and len(set(c) & set(t2_1)) <= 3 and len(set(c) & set(t1_1) & set(t2_1)) == 0), combos_12[2])
    strat_1_tickets = [t1_1, t2_1, t3_1]

    # --- Estrategia 2: Satélite Jerárquico ---
    # T1 es el mejor
    t1_2 = combos_12[0]
    t1_2_set = set(t1_2)
    # T2 mantiene 4 números de T1 y cambia 2 por otros del pool de alta sinergia
    t2_cands = [c for c in combos_12 if len(set(c) & t1_2_set) == 4]
    t2_2 = t2_cands[0] if t2_cands else combos_12[1]
    # T3 mantiene 4 números de T1 (o 3 de T1 y 1 nuevo) con solapamiento controlado 4
    t3_cands = [c for c in combos_12 if len(set(c) & t1_2_set) == 4 and c != t2_2]
    t3_2 = t3_cands[0] if t3_cands else combos_12[2]
    strat_2_tickets = [t1_2, t2_2, t3_2]

    # --- Estrategia 3: 2 Anclas Fijas en los 3 Tickets ---
    top_2_anchors = tuple(ranking[:2])
    rest_pool = [n for n in pool_12 if n not in top_2_anchors]
    combos_rest_4 = [c for c in itertools.combinations(rest_pool, 4) if es_ticket_valido(top_2_anchors + c)]
    if not combos_rest_4: combos_rest_4 = list(itertools.combinations(rest_pool, 4))[:20]
    combos_rest_4.sort(key=lambda sc: sum(borda[n] for n in sc) + 250.0 * sum(cooccur[u][v] for u, v in itertools.combinations(top_2_anchors + sc, 2)), reverse=True)
    t1_3 = tuple(sorted(top_2_anchors + combos_rest_4[0]))
    t2_3 = tuple(sorted(top_2_anchors + combos_rest_4[1]))
    # t3 intenta no solapar tanto con t1 y t2 en las 4 variables
    t3_cand = [c for c in combos_rest_4 if len(set(c) & set(combos_rest_4[0])) <= 2 and len(set(c) & set(combos_rest_4[1])) <= 2]
    t3_3 = tuple(sorted(top_2_anchors + (t3_cand[0] if t3_cand else combos_rest_4[2])))
    strat_3_tickets = [t1_3, t2_3, t3_3]

    # --- Estrategia 4: 3 Anclas Fijas en los 3 Tickets ---
    top_3_anchors = tuple(ranking[:3])
    rest_pool_3 = [n for n in pool_12 if n not in top_3_anchors]
    combos_rest_3 = [c for c in itertools.combinations(rest_pool_3, 3) if es_ticket_valido(top_3_anchors + c)]
    if not combos_rest_3: combos_rest_3 = list(itertools.combinations(rest_pool_3, 3))[:20]
    combos_rest_3.sort(key=lambda sc: sum(borda[n] for n in sc) + 250.0 * sum(cooccur[u][v] for u, v in itertools.combinations(top_3_anchors + sc, 2)), reverse=True)
    t1_4 = tuple(sorted(top_3_anchors + combos_rest_3[0]))
    t2_4 = tuple(sorted(top_3_anchors + combos_rest_3[1]))
    t3_4 = tuple(sorted(top_3_anchors + combos_rest_3[2]))
    strat_4_tickets = [t1_4, t2_4, t3_4]

    strats = [
        ('1. Cobertura Total (Actual: Anti-SPOF, Overlap <= 3, Forzar 12 nums)', strat_1_tickets),
        ('2. Satélite Jerárquico (T1 = Titular, T2 y T3 anclan 4 del T1)', strat_2_tickets),
        ('3. Rueda con 2 Anclas Fijas (Top 2 en los 3 tickets + rotación Pool)', strat_3_tickets),
        ('4. Rueda con 3 Anclas Fijas (Top 3 en los 3 tickets + rotación Pool)', strat_4_tickets),
    ]

    for s_name, t_list in strats:
        hits_per_ticket = [len(set(t) & target_real) for t in t_list]
        max_h = max(hits_per_ticket)
        if max_h >= 3: results[s_name]['3+'] += 1
        if max_h >= 4: results[s_name]['4+'] += 1
        if max_h >= 5: results[s_name]['5+'] += 1
        if max_h >= 4: results[s_name]['premios'] += 1
        results[s_name]['distribucion_hits'].append(hits_per_ticket)

n_eval = len(eval_sorteos)
print("="*95)
print(f"EVALUACIÓN EN {n_eval} SORTEOS TRADICIONALES")
print(f"{'Estrategia':<52} | {'3+ Hits':<9} | {'4+ Hits':<9} | {'5+ Hits':<9} | {'Hits Promedio (T1 / T2 / T3)'}")
print("="*95)
for s_name in results:
    r = results[s_name]
    t1_m = np.mean([h[0] for h in r['distribucion_hits']])
    t2_m = np.mean([h[1] for h in r['distribucion_hits']])
    t3_m = np.mean([h[2] for h in r['distribucion_hits']])
    print(f"{s_name:<52} | {r['3+']:<3} ({r['3+']/n_eval*100:4.1f}%) | {r['4+']:<3} ({r['4+']/n_eval*100:4.1f}%) | {r['5+']:<2} ({r['5+']/n_eval*100:4.1f}%) | T1:{t1_m:.2f}  T2:{t2_m:.2f}  T3:{t3_m:.2f}")
print("="*95)
