with open('generador_cobertura_ml.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Insertar select_balanced_pool_9
pool_func = '''def select_balanced_pool_9(borda_scores, saturated_decades=None, max_per_decade=3):
    """
    Selecciona un Pool de 9 números de máximo score Borda garantizando
    que ninguna decena tenga más de 3 números (max 2 en dec 40 o decenas saturadas),
    asegurando combinaciones viables con dispersión física en 4 decenas.
    """
    ranked = sorted(range(46), key=lambda x: borda_scores[x], reverse=True)
    pool = []
    dec_counts = Counter()
    for n in ranked:
        dec = n // 10
        limit = 2 if dec == 4 else max_per_decade
        if saturated_decades and dec in saturated_decades:
            limit = min(limit, 2)
        if dec_counts[dec] < limit:
            pool.append(n)
            dec_counts[dec] += 1
            if len(pool) == 9:
                break
    if len(pool) < 9:
        for n in ranked:
            if n not in pool:
                pool.append(n)
                if len(pool) == 9: break
    return sorted(pool)

def compute_adaptive_pool('''

code = code.replace('def compute_adaptive_pool(', pool_func)

# 2. En main(), mover el computo de saturated_decades antes de seleccionar los pools y usar select_balanced_pool_9
old_pool_section = '''    ranking_base = sorted(range(46), key=lambda x: borda[x], reverse=True)
    ranking_day = sorted(range(46), key=lambda x: borda_day[x], reverse=True)
    ranking_hib = sorted(range(46), key=lambda x: borda_hib[x], reverse=True)
    
    pool_9_base = sorted(ranking_base[:9])
    pool_9_day = sorted(ranking_day[:9])
    pool_9_hib = sorted(ranking_hib[:9])'''

new_pool_section = '''    # Obtener decena(s) saturada(s) del último sorteo
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

    pool_9_base = select_balanced_pool_9(borda, saturated_decades=saturated_decades)
    pool_9_day = select_balanced_pool_9(borda_day, saturated_decades=saturated_decades)
    pool_9_hib = select_balanced_pool_9(borda_hib, saturated_decades=saturated_decades)'''

code = code.replace(old_pool_section, new_pool_section)

# Eliminar el bloque repetido de saturated_decades que estaba mas abajo
old_repeat_saturated = '''    # Obtener decena(s) saturada(s) del último sorteo
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
            print(f"Decenas saturadas en último sorteo ({last_mod_draw['fecha']}): {[d*10 for d in saturated_decades]}")'''

# Only replace the second occurrence if it appears twice
parts = code.split(old_repeat_saturated)
if len(parts) == 3:
    code = parts[0] + old_repeat_saturated + parts[1] + "# Saturated decades precomputadas arriba" + parts[2]

with open('generador_cobertura_ml.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Actualizado select_balanced_pool_9 con éxito")

