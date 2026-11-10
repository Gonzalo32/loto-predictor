"""
Script para aplicar las 3 mejoras validadas a generador_cobertura_ml.py:
1. select_balanced_pool_9_elite (excepcion de elite para no censurar top 2 en decenas saturadas)
2. Bonus de Super-Resonancia Multi-Modal
3. Regimen dinamico cronologico real sobre fechas de Tradicional
"""
import re

file_path = 'generador_cobertura_ml.py'

with open(file_path, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Reemplazar select_balanced_pool_9 por select_balanced_pool_9 con excepcion de elite
old_pool_func = r'''def select_balanced_pool_9\(borda_scores, saturated_decades=None, max_per_decade=3\):.*?return sorted\(pool\)'''

new_pool_func = '''def select_balanced_pool_9(borda_scores, saturated_decades=None, max_per_decade=3):
    """
    Selecciona un Pool de 9 números de máximo score Borda garantizando
    que ninguna decena tenga más de 3 números (max 2 en dec 40).
    Excepción de Élite: si una decena saturada contiene al Top 1 o Top 2 de la IA,
    permite hasta 3 números para no amputar a candidatos estelares.
    """
    ranked = sorted(range(46), key=lambda x: borda_scores[x], reverse=True)
    elite_numbers = set(ranked[:2])
    pool = []
    dec_counts = Counter()
    for n in ranked:
        dec = n // 10
        limit = 2 if dec == 4 else max_per_decade
        if saturated_decades and dec in saturated_decades:
            is_elite_dec = any((e // 10 == dec) for e in elite_numbers)
            if not is_elite_dec:
                limit = min(limit, 2)
            else:
                limit = 3
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
    return sorted(pool)'''

code, c1 = re.subn(old_pool_func, new_pool_func, code, flags=re.DOTALL)
print(f"Reemplazo 1 (select_balanced_pool_9 con élite): {c1}")

# 2. Reemplazar cálculo de régimen dinámico en main() para que use fechas cronológicas reales de Tradicional
old_regime = r'''    last_5 = draws\[-5:\]
    n_repeats = sum\(len\(set\(last_5\[i\]\['numbers'\]\) & set\(last_5\[i-1\]\['numbers'\]\)\) for i in range\(1, len\(last_5\)\)\)
    avg_repeats = n_repeats / 4\.0
    
    if avg_repeats >= 1\.25:
        regime_desc = "Régimen Caliente / Alta Inercia \(Prioridad Momentum/Recencia\)"
        w_rec = 3\.0
        w_ddm = 1\.0
        w_attn = 3\.2
    elif avg_repeats <= 0\.75:
        regime_desc = "Régimen Frío / Retorno a la Media \(Prioridad Atraso Relativo DDM\)"
        w_rec = 1\.0
        w_ddm = 3\.0
        w_attn = 2\.5
    else:
        regime_desc = "Régimen Balanceado \(Equilibrio Momentum y Atraso\)"
        w_rec = 2\.0
        w_ddm = 1\.8
        w_attn = 3\.0'''

new_regime = '''    # Detección cronológica real de régimen en sorteos de la modalidad objetivo
    trad_draws_all = [d for d in draws if d['modalidad'] == target_mod]
    last_5_trad = trad_draws_all[-5:]
    n_repeats = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
    avg_repeats = n_repeats / 4.0
    
    if avg_repeats >= 1.0:
        regime_desc = "Régimen Caliente / Alta Inercia (Prioridad Momentum/Recencia)"
        w_rec = 2.8
        w_ddm = 1.2
        w_attn = 3.0
    elif avg_repeats <= 0.5:
        regime_desc = "Régimen Frío / Retorno a la Media (Prioridad Atraso Relativo DDM)"
        w_rec = 1.2
        w_ddm = 2.8
        w_attn = 2.5
    else:
        regime_desc = "Régimen Balanceado (Equilibrio Momentum y Atraso)"
        w_rec = 2.0
        w_ddm = 1.8
        w_attn = 2.8'''

code, c2 = re.subn(old_regime, new_regime, code, flags=re.DOTALL)
print(f"Reemplazo 2 (régimen cronológico real): {c2}")

# 3. Insertar Súper-Resonancia Multi-Modal justo después de armar borda
old_borda_end = r'''    borda_ranking = \[n for n, score in sorted\(borda\.items\(\), key=lambda x: x\[1\], reverse=True\)\]'''

new_borda_end = '''    # -------------------------------------------------------------
    # SÚPER-RESONANCIA MULTI-MODAL INMEDIATA
    # Bolillas que salieron en 2 o más modalidades de la fecha previa inmediata
    # -------------------------------------------------------------
    last_sorteo_num = max(d['sorteo'] for d in draws)
    last_date_draws = [d for d in draws if d['sorteo'] == last_sorteo_num]
    res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
    resonance_detected = []
    for n in range(46):
        if res_counts[n] >= 2:
            bonus_res = 18.0 * (res_counts[n] - 1)
            borda[n] += bonus_res
            resonance_detected.append((n, res_counts[n], bonus_res))
            
    if resonance_detected:
        print(f"\\n==================================================")
        print("SÚPER-RESONANCIA MULTI-MODAL DETECTADA (SORTEO PREVIO):")
        for num, app, bon in resonance_detected:
            print(f" -> Número {num:02d}: {app} apariciones en fecha #{last_sorteo_num} (Bonus IA: +{bon:.0f} pts)")
        print(f"==================================================")

    borda_ranking = [n for n, score in sorted(borda.items(), key=lambda x: x[1], reverse=True)]'''

code, c3 = re.subn(old_borda_end, new_borda_end, code, flags=re.DOTALL)
print(f"Reemplazo 3 (Súper-Resonancia Multi-Modal): {c3}")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(code)

print("generador_cobertura_ml.py actualizado con éxito.")

