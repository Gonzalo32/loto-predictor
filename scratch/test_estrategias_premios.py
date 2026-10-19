import csv
import itertools
import numpy as np
from collections import defaultdict, Counter

def load_data(filepath='c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'):
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

# Sorteos de prueba: los últimos 40 sorteos (desde ~3378 hasta 3418)
eval_sorteos = sorteos_unicos[-40:]

print(f"Evaluando en {len(eval_sorteos)} sorteos (desde #{eval_sorteos[0]} hasta #{eval_sorteos[-1]})...")

strategies = {
    '1. Actual (Pool 14, Max Diversity/Orthogonal)': {'3hits': 0, '4hits': 0, '5hits': 0, '6hits': 0, 'premios': 0},
    '2. Concentrado Core-Anchor (Pool 10, Alta Densidad)': {'3hits': 0, '4hits': 0, '5hits': 0, '6hits': 0, 'premios': 0},
    '3. Rueda Combinatoria 4-if-5 (Pool 10, Cobertura Cuartetos)': {'3hits': 0, '4hits': 0, '5hits': 0, '6hits': 0, 'premios': 0},
    '4. Tri-Modal Especializado (T1 Tradicional, T2 Segunda, T3 Core)': {'3hits': 0, '4hits': 0, '5hits': 0, '6hits': 0, 'premios': 0},
}

for s_num in eval_sorteos:
    train_draws = [d for d in draws if d['sorteo'] < s_num]
    real_draws = [d for d in draws if d['sorteo'] == s_num]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in real_draws}
    
    # Calcular scores rápidos y robustos (Frecuencia ventana 60 + Recencia exp + Coocurrencia)
    counts_60 = Counter()
    for d in train_draws[-60:]:
        for n in d['numbers']: counts_60[n] += 1
        
    recency = np.zeros(46)
    for t_idx, d in enumerate(train_draws[-30:]):
        w = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency[n] += w
        
    # Co-ocurrencia exponencial
    cooccur = defaultdict(lambda: defaultdict(float))
    for t_idx, d in enumerate(train_draws[-60:]):
        w = np.exp((t_idx - 60) / 15.0)
        nums = d['numbers']
        for u, v in itertools.combinations(nums, 2):
            cooccur[u][v] += w
            cooccur[v][u] += w
            
    borda = {}
    for n in range(46):
        borda[n] = counts_60[n] * 1.5 + recency[n] * 2.0
    ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)
    
    # Pool 14
    pool_14 = sorted(ranking[:14])
    combos_14 = [c for c in itertools.combinations(pool_14, 6) if es_ticket_valido(c)]
    if not combos_14: combos_14 = list(itertools.combinations(pool_14, 6))[:50]
    
    # Pool 10
    pool_10 = sorted(ranking[:10])
    combos_10 = [c for c in itertools.combinations(pool_10, 6) if es_ticket_valido(c)]
    if not combos_10: combos_10 = list(itertools.combinations(pool_10, 6))[:50]

    # --- Estrategia 1: Actual ---
    core_pool = set(ranking[:8])
    surge_pool = set(pool_14) - core_pool
    v_core = [c for c in combos_14 if sum(1 for n in c if n in core_pool) >= 4] or combos_14
    tA = max(v_core, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur[u][v] for u, v in itertools.combinations(b, 2)))
    
    v_B = [c for c in combos_14 if len(set(c).intersection(set(tA))) <= 3] or combos_14
    tB = max(v_B, key=lambda b: sum(cooccur[u][v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))
    
    borda_C = borda.copy()
    for n in set(tA).union(set(tB)): borda_C[n] *= 0.5
    v_C = [c for c in combos_14 if len(set(c).intersection(set(tA))) <= 3 and len(set(c).intersection(set(tB))) <= 3] or combos_14
    tC = max(v_C, key=lambda b: sum(borda_C[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
    tickets_strat1 = [tA, tB, tC]
    
    # --- Estrategia 2: Concentrado Core-Anchor (Pool 10) ---
    # Top 3 números son anclas fijas (o semi-fijas) o máxima sinergia de pares en Pool 10
    scored_10 = []
    for c in combos_10:
        score_val = sum(borda[n] for n in c) + 300.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2))
        scored_10.append((score_val, c))
    scored_10.sort(key=lambda x: x[0], reverse=True)
    
    t1_s2 = scored_10[0][1]
    # t2 y t3 con overlap 4 (densidad concentrada, no dilute)
    t2_s2 = None
    for _, c in scored_10[1:]:
        if len(set(c).intersection(set(t1_s2))) >= 3 and len(set(c).intersection(set(t1_s2))) <= 4:
            t2_s2 = c
            break
    if not t2_s2: t2_s2 = scored_10[1][1]
    
    t3_s2 = None
    for _, c in scored_10[1:]:
        if c != t1_s2 and c != t2_s2:
            if len(set(c).intersection(set(t1_s2))) >= 3 and len(set(c).intersection(set(t2_s2))) >= 3:
                t3_s2 = c
                break
    if not t3_s2: t3_s2 = scored_10[2][1]
    tickets_strat2 = [t1_s2, t2_s2, t3_s2]

    # --- Estrategia 3: Rueda Combinatoria 4-if-5 sobre Pool 10 ---
    # Maximizar la cobertura de cuartetos (4-subsets) dentro del Pool 10 usando 3 tickets
    all_quads = list(itertools.combinations(pool_10, 4))
    # Greedy max coverage of quads
    cov_quads = set()
    tickets_strat3 = []
    for _ in range(3):
        best_c = None
        best_gain = -1
        for s_val, c in scored_10:
            if c in tickets_strat3: continue
            c_quads = set(itertools.combinations(c, 4))
            gain = len(c_quads - cov_quads) * 1000 + s_val
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c:
            tickets_strat3.append(best_c)
            cov_quads.update(itertools.combinations(best_c, 4))
        else:
            tickets_strat3.append(scored_10[len(tickets_strat3)][1])

    # --- Estrategia 4: Tri-Modal Especializado ---
    # T1 especializado en Tradicional, T2 en Segunda, T3 en Máximo Consenso
    trad_counts = Counter()
    seg_counts = Counter()
    for d in train_draws[-40:]:
        if d['modalidad'] == 'Tradicional':
            for n in d['numbers']: trad_counts[n] += 1
        elif d['modalidad'] == 'Segunda':
            for n in d['numbers']: seg_counts[n] += 1
            
    t1_s4 = max(combos_10, key=lambda c: sum(borda[n] + 15.0 * trad_counts[n] for n in c) + 300.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)))
    t2_s4 = max(combos_10, key=lambda c: sum(borda[n] + 15.0 * seg_counts[n] for n in c) + 300.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)))
    t3_s4 = max(combos_10, key=lambda c: sum(borda[n] for n in c) + 400.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)))
    tickets_strat4 = [t1_s4, t2_s4, t3_s4]

    # Evaluamos en los resultados reales de Tradicional, Segunda y SiempreSale
    # Recordemos: Tradicional y Segunda pagan premios con 4, 5 o 6 aciertos.
    # SiempreSale premia con 5 o 6 aciertos.
    strats_to_eval = [
        ('1. Actual (Pool 14, Max Diversity/Orthogonal)', tickets_strat1),
        ('2. Concentrado Core-Anchor (Pool 10, Alta Densidad)', tickets_strat2),
        ('3. Rueda Combinatoria 4-if-5 (Pool 10, Cobertura Cuartetos)', tickets_strat3),
        ('4. Tri-Modal Especializado (T1 Tradicional, T2 Segunda, T3 Core)', tickets_strat4),
    ]

    for s_name, t_list in strats_to_eval:
        max_h_modal = 0
        premio_obtenido = False
        for t in t_list:
            t_set = set(t)
            h_tr = len(t_set.intersection(real_by_mod.get('Tradicional', set())))
            h_sg = len(t_set.intersection(real_by_mod.get('Segunda', set())))
            h_rv = len(t_set.intersection(real_by_mod.get('Revancha', set())))
            h_ss = len(t_set.intersection(real_by_mod.get('SiempreSale', set())))
            
            # Premio en Tradicional o Segunda: 4, 5, 6
            if h_tr >= 4 or h_sg >= 4:
                premio_obtenido = True
            # Premio en Revancha: 6
            if h_rv >= 6:
                premio_obtenido = True
            # Premio en SiempreSale: 5, 6
            if h_ss >= 5:
                premio_obtenido = True
                
            max_h_modal = max(max_h_modal, h_tr, h_sg, h_rv, h_ss)
            
        if max_h_modal >= 3: strategies[s_name]['3hits'] += 1
        if max_h_modal >= 4: strategies[s_name]['4hits'] += 1
        if max_h_modal >= 5: strategies[s_name]['5hits'] += 1
        if max_h_modal >= 6: strategies[s_name]['6hits'] += 1
        if premio_obtenido: strategies[s_name]['premios'] += 1

print("\n" + "="*95)
print(f"{'Estrategia':<52} | {'3+ Hits':<9} | {'4+ Hits':<9} | {'5+ Hits':<9} | {'Sorteos con Premios'}")
print("="*95)
for s_name in strategies:
    s = strategies[s_name]
    p_prem = (s['premios'] / len(eval_sorteos)) * 100
    print(f"{s_name:<52} | {s['3hits']:<4} ({s['3hits']/len(eval_sorteos)*100:4.1f}%) | {s['4hits']:<4} ({s['4hits']/len(eval_sorteos)*100:4.1f}%) | {s['5hits']:<4} ({s['5hits']/len(eval_sorteos)*100:4.1f}%) | {s['premios']} de {len(eval_sorteos)} ({p_prem:4.1f}%)")
print("="*95)
