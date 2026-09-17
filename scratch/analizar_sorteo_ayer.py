import csv
import json
import os
import sys
import numpy as np

# Set path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Datos de los sorteos 3408 y 3409
sorteo_3408 = {
    'sorteo': 3408, 'fecha': '2026-09-13',
    'Tradicional': [5, 8, 22, 29, 34, 37],
    'Segunda': [4, 20, 25, 31, 39, 42],
    'Revancha': [16, 19, 21, 35, 38, 42],
    'SiempreSale': [11, 15, 33, 36, 43, 44]
}

sorteo_3409 = {
    'sorteo': 3409, 'fecha': '2026-09-16',
    'Tradicional': [13, 14, 16, 19, 20, 29],
    'Segunda': [2, 6, 9, 31, 36, 42],
    'Revancha': [2, 9, 20, 30, 35, 36],
    'SiempreSale': [11, 25, 27, 38, 40, 44]
}

# Cargar predicción de proxima_prediccion.json (del 14 de Septiembre)
pred_path = 'c:/Users/Administrador/Desktop/lot/proxima_prediccion.json'
with open(pred_path, 'r', encoding='utf-8') as f:
    pred = json.load(f)

top_15 = set(pred['top_numeros'])
tickets = pred['tickets']

print("="*60)
print(f"EVALUACIÓN DE LA PREDICCIÓN GENERADA EL: {pred['fecha_prediccion']}")
print(f"JUEGO / MODELO: {pred['juego']}")
print(f"TOP 15 POOL: {sorted(list(top_15))}")
for t in tickets:
    print(f"  - {t['nombre']}: {t['numeros']}")
print("="*60)

def evaluar_sorteo(sorteo_dict):
    print(f"\n--- RESULTS FOR SORTEO #{sorteo_dict['sorteo']} ({sorteo_dict['fecha']}) ---")
    pozo_extra_set = set()
    for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
        nums = sorteo_dict[mod]
        if mod in ['Tradicional', 'Segunda', 'Revancha']:
            pozo_extra_set.update(nums)
        hits_in_pool = [n for n in nums if n in top_15]
        print(f"\nModalidad: {mod}")
        print(f"  Números ganadores: {nums}")
        print(f"  Aciertos en Top 15 Pool ({len(hits_in_pool)}/6): {hits_in_pool}")
        
        for t in tickets:
            t_nums = set(t['numeros'])
            t_hits = t_nums.intersection(set(nums))
            print(f"    {t['nombre']}: {len(t_hits)} aciertos -> {sorted(list(t_hits))}")

    # Pozo Extra evaluation
    pozo_extra_nums = sorted(list(pozo_extra_set))
    hits_extra = [n for n in pozo_extra_nums if n in top_15]
    print(f"\nPozo Extra (Unión de números): {pozo_extra_nums}")
    print(f"  Aciertos en Top 15 Pool ({len(hits_extra)} en total): {hits_extra}")
    for t in tickets:
        t_nums = set(t['numeros'])
        t_hits = t_nums.intersection(pozo_extra_set)
        print(f"    {t['nombre']} en Pozo Extra: {len(t_hits)} aciertos -> {sorted(list(t_hits))}")

evaluar_sorteo(sorteo_3408)
evaluar_sorteo(sorteo_3409)

