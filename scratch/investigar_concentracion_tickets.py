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

draws = load_data()
sorteos_unicos = sorted(list(set(d['sorteo'] for d in draws)))

# Tomamos los últimos 50 sorteos
eval_sorteos = sorteos_unicos[-50:]

rank_hits = Counter()
total_draws_eval = 0

strat_results = {
    'Actual (Wheel Cobertura Total Pool 12, max overlap 3, Anti-SPOF)': {'3': 0, '4': 0, '5': 0, '6': 0, 'tickets_aciertos': []},
    'Anclada (Top 2 Números Ancla en los 3 Tickets + variaciones de Pool 10)': {'3': 0, '4': 0, '5': 0, '6': 0, 'tickets_aciertos': []},
    'Anclada (Top 3 Números Ancla en los 3 Tickets + variaciones de Pool 8)': {'3': 0, '4': 0, '5': 0, '6': 0, 'tickets_aciertos': []},
    'Pool Compacto 8 (Mayor concentración matemática)': {'3': 0, '4': 0, '5': 0, '6': 0, 'tickets_aciertos': []},
    'Jerárquica (T1 = Top 1..6 Puro, T2 = Top 1..4 + 7..8, T3 = Top 1..3 + 5..6 + 9)': {'3': 0, '4': 0, '5': 0, '6': 0, 'tickets_aciertos': []}
}

for s_num in eval_sorteos:
    train_draws = [d for d in draws if d['sorteo'] < s_num]
    real_draws = [d for d in draws if d['sorteo'] == s_num and d['modalidad'] == 'Tradicional']
    if not real_draws:
        continue
    real_nums = set(real_draws[0]['numbers'])
    total_draws_eval += 1
    
    # Calcular ranking de scores robustos (Borda: Ventana 60 + Recencia + Co-ocurrencia)
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
        nums = d['numbers']
        for u, v in itertools.combinations(nums, 2):
            cooccur[u][v] += w
            cooccur[v][u] += w
            
    borda = {}
    for n in range(46):
        borda[n] = counts_60[n] * 1.5 + recency[n] * 2.5
    ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)
    
    # Evaluar dónde caen los aciertos en los primeros 16 puestos
    for rank_idx, n in enumerate(ranking[:16]):
        if n in real_nums:
            rank_hits[rank_idx] += 1
            
    # --- Estrategia 1: Actual (Wheel Cobertura Total Pool 12) ---
    pool_12 = ranking[:12]
    # Simulamos el packaging actual:
    combos_12 = list(itertools.combinations(pool_12, 6))
    combos_12.sort(key=lambda c: sum(borda[n] for n in c) + 200.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)), reverse=True)
    t1_act = combos_12[0]
    t2_act = next((c for c in combos_12 if len(set(c).intersection(set(t1_act))) <= 3), combos_12[1])
    uncovered = set(pool_12) - (set(t1_act) | set(t2_act))
    t3_act = next((c for c in combos_12 if uncovered.issubset(set(c)) and len(set(c) & set(t1_act)) <= 3 and len(set(c) & set(t2_act)) <= 3 and len(set(c) & set(t1_act) & set(t2_act)) == 0), combos_12[2])
    strat_results['Actual (Wheel Cobertura Total Pool 12, max overlap 3, Anti-SPOF)']['tickets_aciertos'].append([len(set(t) & real_nums) for t in [t1_act, t2_act, t3_act]])

    # --- Estrategia 2: Anclada Top 2 en Pool 10 ---
    # Los 2 números de mayor ranking están en los 3 tickets
    anchors_2 = ranking[:2]
    rest_pool_10 = ranking[2:10]
    # Se eligen 4 números restantes para cada ticket buscando máxima sinergia de pares
    sub_combos_4 = list(itertools.combinations(rest_pool_10, 4))
    sub_combos_4.sort(key=lambda sc: sum(borda[n] for n in sc) + 250.0 * sum(cooccur[u][v] for u, v in itertools.combinations(list(anchors_2) + list(sc), 2)), reverse=True)
    t1_anc2 = tuple(sorted(list(anchors_2) + list(sub_combos_4[0])))
    t2_anc2 = tuple(sorted(list(anchors_2) + list(sub_combos_4[1])))
    # El tercer boleto toma una combinación complementaria para cubrir los restantes
    t3_anc2 = tuple(sorted(list(anchors_2) + list(sub_combos_4[2])))
    strat_results['Anclada (Top 2 Números Ancla en los 3 Tickets + variaciones de Pool 10)']['tickets_aciertos'].append([len(set(t) & real_nums) for t in [t1_anc2, t2_anc2, t3_anc2]])

    # --- Estrategia 3: Anclada Top 3 en Pool 8 ---
    anchors_3 = ranking[:3]
    rest_pool_8 = ranking[3:8] # 5 números
    # Hay exactamente C(5, 3) = 10 combinaciones para completar los 3 tickets
    sub_combos_3 = list(itertools.combinations(rest_pool_8, 3))
    sub_combos_3.sort(key=lambda sc: sum(borda[n] for n in sc) + 250.0 * sum(cooccur[u][v] for u, v in itertools.combinations(list(anchors_3) + list(sc), 2)), reverse=True)
    t1_anc3 = tuple(sorted(list(anchors_3) + list(sub_combos_3[0])))
    t2_anc3 = tuple(sorted(list(anchors_3) + list(sub_combos_3[1])))
    t3_anc3 = tuple(sorted(list(anchors_3) + list(sub_combos_3[2])))
    strat_results['Anclada (Top 3 Números Ancla en los 3 Tickets + variaciones de Pool 8)']['tickets_aciertos'].append([len(set(t) & real_nums) for t in [t1_anc3, t2_anc3, t3_anc3]])

    # --- Estrategia 4: Pool Compacto 8 (C(8,6) = 28 combos, elegimos los 3 mejores por sinergia) ---
    pool_8 = ranking[:8]
    combos_8 = list(itertools.combinations(pool_8, 6))
    combos_8.sort(key=lambda c: sum(borda[n] for n in c) + 300.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)), reverse=True)
    t1_p8 = combos_8[0]
    t2_p8 = combos_8[1]
    t3_p8 = combos_8[2]
    strat_results['Pool Compacto 8 (Mayor concentración matemática)']['tickets_aciertos'].append([len(set(t) & real_nums) for t in [t1_p8, t2_p8, t3_p8]])

    # --- Estrategia 5: Jerárquica Directa ---
    # T1 = Ranking 0..5 (El Top 6 absoluto)
    # T2 = Ranking [0, 1, 2, 3, 6, 7] (Cambia 2 del fondo por los puestos 7 y 8)
    # T3 = Ranking [0, 1, 4, 5, 8, 9] (Cambia otros 2 por puestos 9 y 10)
    t1_jer = tuple(sorted(ranking[:6]))
    t2_jer = tuple(sorted(ranking[:4] + ranking[6:8]))
    t3_jer = tuple(sorted(ranking[:2] + ranking[4:6] + ranking[8:10]))
    strat_results['Jerárquica (T1 = Top 1..6 Puro, T2 = Top 1..4 + 7..8, T3 = Top 1..3 + 5..6 + 9)']['tickets_aciertos'].append([len(set(t) & real_nums) for t in [t1_jer, t2_jer, t3_jer]])

# Computar métricas
print("="*80)
print(f"ANÁLISIS DE EFICACIA EN LOS ÚLTIMOS {total_draws_eval} SORTEOS TRADICIONALES")
print("="*80)
print("\n[1] DISTRIBUCIÓN DE ACIERTOS SEGÚN EL RANGO EN EL RANKING DEL MODELO:")
print(f"{'Rango Ranking':<20} | {'Aciertos Totales':<18} | {'% de los Aciertos'}")
print("-" * 55)
total_hits_all_ranks = sum(rank_hits.values())
for group_name, r_range in [("Top 1 a 3", range(3)), ("Top 4 a 6", range(3, 6)), ("Top 7 a 9", range(6, 9)), ("Top 10 a 12", range(9, 12)), ("Puestos 13 a 15", range(12, 15))]:
    g_hits = sum(rank_hits[r] for r in r_range)
    pct = (g_hits / total_hits_all_ranks) * 100 if total_hits_all_ranks else 0
    print(f"{group_name:<20} | {g_hits:<18} | {pct:.1f}%")

print("\n" + "="*80)
print("[2] COMPARATIVA DE ESTRATEGIAS DE EMPAQUETADO (MAX ACIERTOS POR SORTEO):")
print(f"{'Estrategia':<48} | {'3+ Hits':<9} | {'4+ Hits':<9} | {'5+ Hits':<9} | {'Promedio T1':<12}")
print("-" * 92)

for s_name, data in strat_results.items():
    tickets_hist = data['tickets_aciertos']
    max_hits_list = [max(t_hits) for t_hits in tickets_hist]
    t1_hits_list = [t_hits[0] for t_hits in tickets_hist]
    
    c3 = sum(1 for m in max_hits_list if m >= 3)
    c4 = sum(1 for m in max_hits_list if m >= 4)
    c5 = sum(1 for m in max_hits_list if m >= 5)
    avg_t1 = np.mean(t1_hits_list)
    
    pct3 = (c3 / total_draws_eval) * 100
    pct4 = (c4 / total_draws_eval) * 100
    pct5 = (c5 / total_draws_eval) * 100
    
    print(f"{s_name[:48]:<48} | {c3:<3} ({pct3:4.1f}%) | {c4:<3} ({pct4:4.1f}%) | {c5:<2} ({pct5:4.1f}%) | {avg_t1:.2f} aciertos")
print("="*80)
