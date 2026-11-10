"""
Script para aplicar las mejoras validadas a generador_cobertura_ml.py:
1. Filtro Anti-Congestion (minimo 4 decenas distintas)
2. Bloqueo de Pares Antagonicos (Lift < 0.40)
3. Modelado Dinamico por Dia de la Semana (Miercoles vs Domingo)
4. Portfolio Especializado de 3 Boletos (1 cupon habitual, 9 numeros unicos)
"""
import re

file_path = 'generador_cobertura_ml.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Reemplazar es_ticket_valido agregando compute_antagonist_pairs y nuevos parametros
old_func_pattern = r"def es_ticket_valido\(ticket, recent_numbers=None, saturated_decades=None\):.*?return True"

new_func = '''def compute_antagonist_pairs(train_draws, min_expected=10.0, max_lift=0.40):
    """
    Identifica pares que tienen una frecuencia observada significativamente inferior
    a la esperada por azar en la modalidad Tradicional (lift < 0.40), evitando
    combinaciones antagónicas que casi nunca salen juntas.
    """
    trad_draws = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    T = len(trad_draws)
    if T < 50:
        return set()
    freq = Counter(n for d in trad_draws for n in d['numbers'])
    pairs = Counter(tuple(sorted([u, v])) for d in trad_draws for u, v in itertools.combinations(d['numbers'], 2))
    antagonist_pairs = set()
    for u, v in itertools.combinations(range(46), 2):
        p_exp = (freq[u] * freq[v]) / T
        obs = pairs[tuple(sorted([u, v]))]
        lift = obs / p_exp if p_exp > 0 else 1.0
        if p_exp >= min_expected and lift < max_lift:
            antagonist_pairs.add(tuple(sorted([u, v])))
    return antagonist_pairs

def es_ticket_valido(ticket, recent_numbers=None, saturated_decades=None, antagonist_pairs=None, min_distinct_decades=4):
    ticket = sorted(ticket)
    # 1. Consecutividad: máximo dos parejas consecutivas
    gaps_le_1 = sum(1 for i in range(1, len(ticket)) if ticket[i] - ticket[i-1] <= 1)
    if gaps_le_1 > 2:
        return False
    # 2. Spread: entre 15 y 45
    spread = ticket[5] - ticket[0]
    if spread < 15 or spread > 45:
        return False
    # 3. Decenas: máximo 3 números por decena (y máximo 2 en la decena del 40 ya que sólo hay 6 números: 40-45)
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(count > 3 for count in decenas.values()):
        return False
    if decenas[4] > 2:
        return False
    # 3b. Anti-Congestión: exigir que los 6 números toquen al menos min_distinct_decades decenas distintas
    if len(decenas) < min_distinct_decades:
        return False
    # 4. Paridad: entre 1 y 5 números pares (óptimo 2 a 4)
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares == 0 or pares == 6:
        return False
    # 5. Partición caliente/frío: exactamente 1 a 5 números calientes
    if recent_numbers is not None:
        recent_count = sum(1 for n in ticket if n in recent_numbers)
        if recent_count < 1 or recent_count > 5:
            return False
    # 6. Decade Saturation Deflation (DSD) Filter
    if saturated_decades:
        for dec in saturated_decades:
            if decenas[dec] > 2:
                return False
    # 7. Bloqueo de Pares Antagónicos (Lift < 0.40)
    if antagonist_pairs:
        for u, v in itertools.combinations(ticket, 2):
            if tuple(sorted([u, v])) in antagonist_pairs:
                return False
    return True'''

content, count1 = re.subn(old_func_pattern, new_func, content, flags=re.DOTALL)
print(f"Reemplazo 1 (es_ticket_valido y compute_antagonist_pairs): {count1} ocurrencias")

# 2. Reemplazar la seccion de generacion de boletos en main()
old_main_gen_pattern = r"# Pool Dinámico Adaptativo Compactado.*?# Export to proxima_prediccion\.json"

new_main_gen = '''# -------------------------------------------------------------
    # MODELADO DINÁMICO POR DÍA DE LA SEMANA (MIÉRCOLES VS DOMINGO)
    # -------------------------------------------------------------
    last_draw = draws[-1]
    last_draw_date = last_draw['fecha']
    last_weekday = datetime.strptime(last_draw_date, '%Y-%m-%d').weekday()
    target_weekday = 2 if last_weekday == 6 else 6
    weekday_name = "Miércoles" if target_weekday == 2 else "Domingo"
    
    same_day_draws = [d for d in draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
    
    borda_day = defaultdict(float)
    borda_hib = defaultdict(float)
    for n in range(46):
        borda_day[n] = borda[n] + 16.0 * counts_day[n]
        borda_hib[n] = borda[n] + 8.0 * counts_day[n]
        
    ranking_base = sorted(range(46), key=lambda x: borda[x], reverse=True)
    ranking_day = sorted(range(46), key=lambda x: borda_day[x], reverse=True)
    ranking_hib = sorted(range(46), key=lambda x: borda_hib[x], reverse=True)
    
    pool_9_base = sorted(ranking_base[:9])
    pool_9_day = sorted(ranking_day[:9])
    pool_9_hib = sorted(ranking_hib[:9])
    
    print(f"\\n==================================================")
    print(f"MODELO ESPECIALIZADO POR DÍA ({weekday_name.upper()}):")
    print(f" -> Pool Base Tradicional (9): {pool_9_base}")
    print(f" -> Pool Día {weekday_name} (9): {pool_9_day}")
    print(f"==================================================")

    # Precomputar pares antagónicos para filtrado físico
    antagonist_pairs = compute_antagonist_pairs(draws)
    print(f" -> Pares antagónicos bloqueados (Lift < 0.40): {len(antagonist_pairs)}")

    # Generar números calientes recientes de los últimos 5 sorteos secuenciales
    recent_numbers = set()
    for d in draws[-5:]:
        recent_numbers.update(d['numbers'])
        
    # Obtener decena(s) saturada(s) del último sorteo
    saturated_decades = set()
    last_mod_draw = None
    for d in reversed(draws):
        if d['modalidad'] == target_mod:
            last_mod_draw = d
            break
    if last_mod_draw:
        dec_counts = Counter(num // 10 for num in last_mod_draw['numbers'])
        for dec, count in dec_counts.items():
            if count >= 3:
                saturated_decades.add(dec)
        if saturated_decades:
            print(f"Decenas saturadas en último sorteo ({last_mod_draw['fecha']}): {[d*10 for d in saturated_decades]}")

    # Filtrar combinaciones con Filtro Anti-Congestión (4 decenas) y Bloqueo Antagónico
    def get_viable_combos(pool):
        combos = list(itertools.combinations(pool, 6))
        # 1. Intentar con 4 decenas distintas y pares antagónicos
        v = [c for c in combos if es_ticket_valido(c, recent_numbers, saturated_decades, antagonist_pairs, min_distinct_decades=4)]
        if len(v) < 3:
            v = [c for c in combos if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=4)]
        if len(v) < 3:
            v = [c for c in combos if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
        if len(v) < 3:
            v = combos[:10]
        return v

    viable_base = get_viable_combos(pool_9_base)
    viable_day = get_viable_combos(pool_9_day)
    viable_hib = get_viable_combos(pool_9_hib)

    print(f"Combinaciones viables: Base={len(viable_base)}, Día={len(viable_day)}, Híbrido={len(viable_hib)}")

    # Generar ternas óptimas
    t_base = select_optimal_wheel_triplet(viable_base, pool_9_base, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_day = select_optimal_wheel_triplet(viable_day, pool_9_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_hib = select_optimal_wheel_triplet(viable_hib, pool_9_hib, borda_hib, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)

    # -------------------------------------------------------------
    # PORTFOLIO ESPECIALIZADO DE 3 BOLETOS (1 CUPÓN HABITUAL)
    # Boleto 1: Cazador Tradicional (Modelo A)
    # Boleto 2: Especialista Día de la Semana (Modelo B)
    # Boleto 3: Consenso / Cobertura Híbrida (Modelo C)
    # -------------------------------------------------------------
    b1 = t_base[0]
    b2 = t_day[0] if t_day[0] != b1 else t_day[1]
    b3 = t_hib[0] if (t_hib[0] != b1 and t_hib[0] != b2) else t_base[1]
    tickets_optimos = [b1, b2, b3]

    total_unique_numbers = sorted(list(set(b1) | set(b2) | set(b3)))

    # Simular probabilidad de éxito conjunta
    prob_exito = estimar_probabilidad_exito_conjunto(tickets_optimos, probs_t)
    
    individual_probs = []
    for b in tickets_optimos:
        c_probs = [probs_t[n] for n in sorted(b)]
        h3, h4, h5, h6 = prob_hits_poisson_binomial_flat(c_probs)
        individual_probs.append((h3 + h4 + h5 + h6) * 100)
    
    print("\\n==================================================")
    print("PORTFOLIO ESPECIALIZADO DE 3 BOLETOS (QUINI 6):")
    print(f"Total números únicos jugados en el cupón: {len(total_unique_numbers)} {total_unique_numbers}")
    print("==================================================")
    ticket_names = [
        f"SUPER TICKET 1 [ESPECIALISTA TRADICIONAL - MÁXIMO POZO]",
        f"SUPER TICKET 2 [ESPECIALISTA DÍA {weekday_name.upper()} - ARRASTRE MULTI-MODAL]",
        f"SUPER TICKET 3 [CONSENSO HÍBRIDO - RED DE COBERTURA POZO EXTRA]"
    ]
    tickets_json = []
    for idx, ticket in enumerate(tickets_optimos):
        t_sorted = [int(x) for x in sorted(ticket)]
        t_name = ticket_names[idx]
        print(f"Boleto {idx+1:02d} [{t_name}]: {t_sorted} | Prob. Indiv. (3+): {individual_probs[idx]:.2f}%")
        tickets_json.append({
            "nombre": t_name,
            "numeros": [int(x) for x in t_sorted]
        })
    print("==================================================")
    print(f"PROBABILIDAD ACUMULADA CONJUNTA (3+ aciertos en al menos 1 boleto): {prob_exito:.2f}%")
    print(f"Desempeño relativo frente al azar: {(prob_exito / 5.48):.2f}x superior")
    print("==================================================")

    top_pool = total_unique_numbers

    # Export to proxima_prediccion.json'''

content, count2 = re.subn(old_main_gen_pattern, new_main_gen, content, flags=re.DOTALL)
print(f"Reemplazo 2 (main generation logic): {count2} ocurrencias")

if count1 == 1 and count2 == 1:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("[EXITO] generador_cobertura_ml.py actualizado correctamente.")
else:
    print("[ERROR] No se pudieron aplicar todas las sustituciones.")

