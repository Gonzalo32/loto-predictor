import csv
import itertools
import numpy as np
from collections import defaultdict, Counter
import time

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

# Evaluamos en los últimos 50 sorteos
eval_sorteos = sorteos_unicos[-50:]

print("="*95)
print(f"EXPERIMENTO EXHAUSTIVO DE EMPAQUETADO Y PREMIOS EN {len(eval_sorteos)} SORTEOS REALES")
print(f"Desde Sorteo #{eval_sorteos[0]} hasta #{eval_sorteos[-1]}")
print("="*95)

results = {
    '1. Actual (Pool 14, Max Dispersion Overlap<=3)': {'3h': 0, '4h': 0, '5h': 0, 'premios_efectivos': 0, 'pe_6h': 0, 'max_hits_list': []},
    '2. Soft Overlap (Pool 14, Overlap 4 Permitido)': {'3h': 0, '4h': 0, '5h': 0, 'premios_efectivos': 0, 'pe_6h': 0, 'max_hits_list': []},
    '3. Core-Anchor (Pool 10, Anclas Clave Top 3)': {'3h': 0, '4h': 0, '5h': 0, 'premios_efectivos': 0, 'pe_6h': 0, 'max_hits_list': []},
    '4. Especializado por Modalidad (T1:Trad, T2:Seg, T3:Core)': {'3h': 0, '4h': 0, '5h': 0, 'premios_efectivos': 0, 'pe_6h': 0, 'max_hits_list': []},
    '5. Rueda Reducida 4-if-5 (Pool 10, Máxima Densidad)': {'3h': 0, '4h': 0, '5h': 0, 'premios_efectivos': 0, 'pe_6h': 0, 'max_hits_list': []},
    '6. Hybrid Dual-Core (T1: Core Top 6, T2: Core B, T3: Sinergia)': {'3h': 0, '4h': 0, '5h': 0, 'premios_efectivos': 0, 'pe_6h': 0, 'max_hits_list': []},
}

for s_num in eval_sorteos:
    train_draws = [d for d in draws if d['sorteo'] < s_num]
    real_draws = [d for d in draws if d['sorteo'] == s_num]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in real_draws}
    trad_real = real_by_mod.get('Tradicional', set())
    seg_real = real_by_mod.get('Segunda', set())
    rev_real = real_by_mod.get('Revancha', set())
    ss_real = real_by_mod.get('SiempreSale', set())
    pe_real = trad_real.union(seg_real).union(rev_real)

    # 1. Puntuación ML / Estadística rápida
    # Frecuencia en ventana 60
    counts_60 = Counter()
    for d in train_draws[-60:]:
        for n in d['numbers']: counts_60[n] += 1
        
    # Recencia exponencial
    recency = np.zeros(46)
    for t_idx, d in enumerate(train_draws[-30:]):
        w = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency[n] += w

    # Delays actuales
    last_seen = {}
    for idx_d, d in enumerate(train_draws):
        for n in d['numbers']: last_seen[n] = idx_d
    delays = {n: len(train_draws) - 1 - last_seen.get(n, 0) for n in range(46)}
    # Momentum: penalizar muy fríos (>25 sin salir), bonificar ciclo óptimo (3..12)
    momentum = {}
    for n in range(46):
        d = delays[n]
        if 3 <= d <= 12: momentum[n] = 2.0
        elif d > 25: momentum[n] = -1.5
        else: momentum[n] = 0.5

    # Coocurrencia exponencial
    cooccur = defaultdict(lambda: defaultdict(float))
    for t_idx, d in enumerate(train_draws[-60:]):
        w = np.exp((t_idx - 60) / 15.0)
        nums = d['numbers']
        for u, v in itertools.combinations(nums, 2):
            cooccur[u][v] += w
            cooccur[v][u] += w

    # Frecuencia específica por modalidad
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

    # Borda Global
    borda = {}
    for n in range(46):
        borda[n] = counts_60[n] * 1.5 + recency[n] * 2.0 + momentum[n] * 1.2
    ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)

    pool_14 = sorted(ranking[:14])
    combos_14 = [c for c in itertools.combinations(pool_14, 6) if es_ticket_valido(c)]
    if not combos_14: combos_14 = list(itertools.combinations(pool_14, 6))[:50]

    pool_10 = sorted(ranking[:10])
    combos_10 = [c for c in itertools.combinations(pool_10, 6) if es_ticket_valido(c)]
    if not combos_10: combos_10 = list(itertools.combinations(pool_10, 6))[:50]

    pool_8 = sorted(ranking[:8])
    combos_8 = [c for c in itertools.combinations(pool_8, 6) if es_ticket_valido(c)]
    if not combos_8: combos_8 = list(itertools.combinations(pool_8, 6))

    # ==============================================================
    # 1. ACTUAL (Pool 14, Max Dispersion Overlap <= 3)
    # ==============================================================
    core_pool = set(ranking[:8])
    surge_pool = set(pool_14) - core_pool
    v_core = [c for c in combos_14 if sum(1 for n in c if n in core_pool) >= 4] or combos_14
    tA_1 = max(v_core, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur[u][v] for u, v in itertools.combinations(b, 2)))
    v_B = [c for c in combos_14 if len(set(c).intersection(set(tA_1))) <= 3] or combos_14
    tB_1 = max(v_B, key=lambda b: sum(cooccur[u][v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))
    borda_C = borda.copy()
    for n in set(tA_1).union(set(tB_1)): borda_C[n] *= 0.5
    v_C = [c for c in combos_14 if len(set(c).intersection(set(tA_1))) <= 3 and len(set(c).intersection(set(tB_1))) <= 3] or combos_14
    tC_1 = max(v_C, key=lambda b: sum(borda_C[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
    strat_1 = [tA_1, tB_1, tC_1]

    # ==============================================================
    # 2. SOFT OVERLAP (Pool 14, Overlap 4 Permitido, sin castigar Borda en C)
    # ==============================================================
    # Permite que combinaciones ganadoras densas de 4 números coexistan en los tickets
    scored_14 = []
    for c in combos_14:
        s = sum(borda[n] for n in c) + 350.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2))
        scored_14.append((s, c))
    scored_14.sort(key=lambda x: x[0], reverse=True)

    tA_2 = scored_14[0][1]
    # tB con overlap hasta 4 con tA (mantiene el cluster fuerte pero varía 2 números)
    tB_2 = None
    for _, c in scored_14[1:]:
        ov = len(set(c).intersection(set(tA_2)))
        if 2 <= ov <= 4:
            tB_2 = c
            break
    if not tB_2: tB_2 = scored_14[1][1]

    # tC con overlap hasta 4 con A o B pero con variación de rotadores
    tC_2 = None
    for _, c in scored_14[1:]:
        if c != tA_2 and c != tB_2:
            ovA = len(set(c).intersection(set(tA_2)))
            ovB = len(set(c).intersection(set(tB_2)))
            if ovA <= 4 and ovB <= 4 and (ovA >= 2 or ovB >= 2):
                tC_2 = c
                break
    if not tC_2: tC_2 = scored_14[2][1]
    strat_2 = [tA_2, tB_2, tC_2]

    # ==============================================================
    # 3. CORE-ANCHOR (Pool 10, Anclas Clave Top 3 fijas o semi-fijas)
    # ==============================================================
    anchors_3 = set(ranking[:3]) # Top 3 números con mayor score
    # Ticket A: contiene las 3 anclas + mejores 3 de ranking 3..9
    # Ticket B: contiene 2 de las 3 anclas + mejores 4
    # Ticket C: contiene 2 de las 3 anclas (otra combinación) + mejores 4
    v_with_anchors = [c for c in combos_10 if len(set(c).intersection(anchors_3)) >= 2]
    if not v_with_anchors: v_with_anchors = combos_10
    scored_anchor = []
    for c in v_with_anchors:
        s = sum(borda[n] for n in c) + 300.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2))
        scored_anchor.append((s, c))
    scored_anchor.sort(key=lambda x: x[0], reverse=True)
    tA_3 = scored_anchor[0][1]
    tB_3 = scored_anchor[1][1] if len(scored_anchor) > 1 else tA_3
    tC_3 = scored_anchor[2][1] if len(scored_anchor) > 2 else tB_3
    strat_3 = [tA_3, tB_3, tC_3]

    # ==============================================================
    # 4. ESPECIALIZADO POR MODALIDAD
    # T1 especializado en Tradicional (pares + frecuencias de Trad)
    # T2 especializado en Segunda (pares + frecuencias de Seg)
    # T3 especializado en Consenso / Siempre Sale
    # ==============================================================
    # T1 Tradicional
    tA_4 = max(combos_14, key=lambda c: sum(borda[n] + 18.0 * trad_counts[n] for n in c) + 350.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)))
    # T2 Segunda (optimizado con sesgo de Segunda)
    tB_4 = max(combos_14, key=lambda c: sum(borda[n] + 18.0 * seg_counts[n] for n in c) + 350.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)))
    # T3 Consenso Siempre Sale / Cobertura
    tC_4 = max(combos_14, key=lambda c: sum(borda[n] + 12.0 * ss_counts[n] for n in c) + 400.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)))
    strat_4 = [tA_4, tB_4, tC_4]

    # ==============================================================
    # 5. RUEDA REDUCIDA 4-if-5 (Pool 10, Máxima Cobertura de Cuartetos)
    # ==============================================================
    # Selecciona 3 boletos de Pool 10 que maximizan la cobertura de cuartetos
    cov_quads = set()
    strat_5 = []
    candidates_5 = []
    for c in combos_10:
        s = sum(borda[n] for n in c) + 250.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2))
        candidates_5.append((s, c))
    candidates_5.sort(key=lambda x: x[0], reverse=True)
    
    for _ in range(3):
        best_c = None
        best_score = -1e9
        for s, c in candidates_5:
            if c in strat_5: continue
            quads_c = set(itertools.combinations(c, 4))
            new_quads = len(quads_c - cov_quads)
            val = new_quads * 500.0 + s
            if val > best_score:
                best_score = val
                best_c = c
        if best_c:
            strat_5.append(best_c)
            cov_quads.update(itertools.combinations(best_c, 4))
        else:
            strat_5.append(candidates_5[len(strat_5)][1])

    # ==============================================================
    # 6. HYBRID DUAL-CORE
    # Ticket 1: El mejor boleto de los Top 8 números (máxima densidad)
    # Ticket 2: El segundo mejor boleto de Top 8 con 4 o 5 números en común
    # Ticket 3: Mejor boleto cruzando Top 4 con Pool 9..14 (cobertura externa)
    # ==============================================================
    if combos_8:
        scored_8 = [(sum(borda[n] for n in c) + 300.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)), c) for c in combos_8]
        scored_8.sort(key=lambda x: x[0], reverse=True)
        tA_6 = scored_8[0][1]
        tB_6 = scored_8[1][1] if len(scored_8) > 1 else scored_8[0][1]
    else:
        tA_6 = combos_10[0]
        tB_6 = combos_10[1]
    # Ticket C cruza top 4 anclas con los mejores de surge pool
    top4_anchors = set(ranking[:4])
    v_surge_cross = [c for c in combos_14 if len(set(c).intersection(top4_anchors)) >= 3 and any(n in ranking[8:14] for n in c)]
    if not v_surge_cross: v_surge_cross = combos_14
    tC_6 = max(v_surge_cross, key=lambda c: sum(borda[n] for n in c) + 200.0 * sum(cooccur[u][v] for u, v in itertools.combinations(c, 2)))
    strat_6 = [tA_6, tB_6, tC_6]

    # --- EVALUACIÓN DE LAS 6 ESTRATEGIAS ---
    all_strats = [
        ('1. Actual (Pool 14, Max Dispersion Overlap<=3)', strat_1),
        ('2. Soft Overlap (Pool 14, Overlap 4 Permitido)', strat_2),
        ('3. Core-Anchor (Pool 10, Anclas Clave Top 3)', strat_3),
        ('4. Especializado por Modalidad (T1:Trad, T2:Seg, T3:Core)', strat_4),
        ('5. Rueda Reducida 4-if-5 (Pool 10, Máxima Densidad)', strat_5),
        ('6. Hybrid Dual-Core (T1: Core Top 6, T2: Core B, T3: Sinergia)', strat_6),
    ]

    for name, t_list in all_strats:
        max_h_modal = 0
        premio_cobrado = False
        max_pe = 0
        
        for t in t_list:
            t_set = set(t)
            h_tr = len(t_set.intersection(trad_real))
            h_sg = len(t_set.intersection(seg_real))
            h_rv = len(t_set.intersection(rev_real))
            h_ss = len(t_set.intersection(ss_real))
            h_pe = len(t_set.intersection(pe_real))
            
            # Condiciones oficiales de premio en Quini 6:
            # - Tradicional: 4, 5, 6
            # - Segunda: 4, 5, 6
            # - Revancha: 6
            # - Siempre Sale: 5, 6
            if h_tr >= 4 or h_sg >= 4 or h_rv >= 6 or h_ss >= 5:
                premio_cobrado = True
                
            max_h_modal = max(max_h_modal, h_tr, h_sg, h_rv, h_ss)
            max_pe = max(max_pe, h_pe)

        results[name]['max_hits_list'].append(max_h_modal)
        if max_h_modal >= 3: results[name]['3h'] += 1
        if max_h_modal >= 4: results[name]['4h'] += 1
        if max_h_modal >= 5: results[name]['5h'] += 1
        if premio_cobrado: results[name]['premios_efectivos'] += 1
        if max_pe >= 6: results[name]['pe_6h'] += 1

print("\n" + "="*110)
print(f"{'Estrategia':<56} | {'Avg Hit':<8} | {'3+ Hits':<10} | {'4+ Hits (Premios)':<18} | {'5+ Hits':<8} | {'Premio Cobrado %'}")
print("="*110)
N = len(eval_sorteos)
for name in results:
    r = results[name]
    avg_h = np.mean(r['max_hits_list'])
    p3 = (r['3h'] / N) * 100
    p4 = (r['4h'] / N) * 100
    p5 = (r['5h'] / N) * 100
    prem_pct = (r['premios_efectivos'] / N) * 100
    print(f"{name:<56} | {avg_h:<8.2f} | {r['3h']} ({p3:4.1f}%) | {r['4h']} ({p4:4.1f}%)          | {r['5h']} ({p5:4.1f}%) | {r['premios_efectivos']}/{N} ({prem_pct:4.1f}%)")
print("="*110)
