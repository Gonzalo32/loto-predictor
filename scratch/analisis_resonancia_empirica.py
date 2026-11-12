import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from generador_cobertura_ml import load_data

draws = load_data()
sorteos = sorted(list(set(d['sorteo'] for d in draws)))

# For every date t, count appearances of each ball across all 4 modalities
# Then check if that ball appears in date t+1 (in Tradicional, and in Any modality)

total_cases_2 = 0
hit_trad_2 = 0
hit_any_2 = 0

total_cases_3 = 0
hit_trad_3 = 0
hit_any_3 = 0

total_cases_1 = 0
hit_trad_1 = 0
hit_any_1 = 0

# Base rate for any ball
total_ball_opportunities = 0
ball_trad_hits = 0
ball_any_hits = 0

for i in range(len(sorteos) - 1):
    s_curr = sorteos[i]
    s_next = sorteos[i+1]
    
    draws_curr = [d for d in draws if d['sorteo'] == s_curr]
    draws_next = [d for d in draws if d['sorteo'] == s_next]
    
    counts_curr = Counter(n for d in draws_curr for n in d['numbers'])
    
    trad_next_balls = set(n for d in draws_next if d['modalidad'] == 'Tradicional' for n in d['numbers'])
    
    any_next_balls = set(n for d in draws_next for n in d['numbers'])
    
    for n in range(46):
        c = counts_curr.get(n, 0)
        total_ball_opportunities += 1
        if n in trad_next_balls:
            ball_trad_hits += 1
        if n in any_next_balls:
            ball_any_hits += 1
            
        if c == 1:
            total_cases_1 += 1
            if n in trad_next_balls:
                hit_trad_1 += 1
            if n in any_next_balls:
                hit_any_1 += 1
        elif c == 2:
            total_cases_2 += 1
            if n in trad_next_balls:
                hit_trad_2 += 1
            if n in any_next_balls:
                hit_any_2 += 1
        elif c >= 3:
            total_cases_3 += 1
            if n in trad_next_balls:
                hit_trad_3 += 1
            if n in any_next_balls:
                hit_any_3 += 1

print("==================================================================")
print("ANÁLISIS HISTÓRICO EMPÍRICO DE SÚPER-RESONANCIA (659 FECHAS)")
print("==================================================================")
print(f"Línea de base aleatoria (probabilidad de cualquier bolilla de salir):")
print(f" -> En Tradicional (6 de 46) : {ball_trad_hits / total_ball_opportunities * 100:.2f}% (Teórico: 13.04%)")
print(f" -> En Cualquier Mod. (20-22): {ball_any_hits / total_ball_opportunities * 100:.2f}% (Teórico: ~44%)")
print("\nComportamiento cuando una bolilla sale 1 VEZ en la fecha previa:")
print(f" -> Casos analizados: {total_cases_1}")
print(f" -> Repite en Tradicional fecha siguiente: {hit_trad_1 / total_cases_1 * 100:.2f}% (Lift: { (hit_trad_1 / total_cases_1) / (ball_trad_hits / total_ball_opportunities):.2f}x)")
print(f" -> Repite en Cualquier Modalidad fecha siguiente: {hit_any_1 / total_cases_1 * 100:.2f}%")

print("\nComportamiento cuando una bolilla sale 2 VECES en la fecha previa (RESONANCIA):")
print(f" -> Casos analizados: {total_cases_2}")
print(f" -> Repite en Tradicional fecha siguiente: {hit_trad_2 / total_cases_2 * 100:.2f}% (Lift: { (hit_trad_2 / total_cases_2) / (ball_trad_hits / total_ball_opportunities):.2f}x)")
print(f" -> Repite en Cualquier Modalidad fecha siguiente: {hit_any_2 / total_cases_2 * 100:.2f}%")

print("\nComportamiento cuando una bolilla sale 3+ VECES en la fecha previa (SÚPER-RESONANCIA):")
print(f" -> Casos analizados: {total_cases_3}")
print(f" -> Repite en Tradicional fecha siguiente: {hit_trad_3 / total_cases_3 * 100:.2f}% (Lift: { (hit_trad_3 / total_cases_3) / (ball_trad_hits / total_ball_opportunities):.2f}x)")
print(f" -> Repite en Cualquier Modalidad fecha siguiente: {hit_any_3 / total_cases_3 * 100:.2f}%")
print("==================================================================")
