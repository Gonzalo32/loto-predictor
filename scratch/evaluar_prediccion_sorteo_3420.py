import json
import os
import sys

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pred_path = os.path.join(WORKSPACE_DIR, 'proxima_prediccion.json')

with open(pred_path, 'r', encoding='utf-8') as f:
    pred = json.load(f)

top_pool = set(pred['top_numeros'])
tickets = pred['tickets']

print("==================================================")
print("PREDICCIÓN GENERADA EL:", pred['fecha_prediccion'])
print("Modelo:", pred['juego'])
print("Top Pool (12 números):", sorted(list(top_pool)))
print("--------------------------------------------------")
for t in tickets:
    print(f"  {t['nombre']}: {t['numeros']}")
print("==================================================\n")

# Resultados oficiales Sorteo 3420 (25/10/2026 - Domingo pasado)
sorteo_3420 = {
    'sorteo': 3420,
    'fecha': '2026-10-25',
    'Tradicional': [0, 2, 3, 8, 19, 39],
    'Segunda': [8, 25, 30, 36, 43, 45],
    'Revancha': [5, 22, 23, 29, 41, 43],
    'SiempreSale': [0, 13, 22, 24, 40, 43]
}

pozo_extra_set = set(sorteo_3420['Tradicional'] + sorteo_3420['Segunda'] + sorteo_3420['Revancha'])

print(f"EVALUACIÓN DETALLADA - SORTEO #{sorteo_3420['sorteo']} ({sorteo_3420['fecha']})")
print("="*60)

for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
    nums = sorteo_3420[mod]
    hits_pool = [n for n in nums if n in top_pool]
    print(f"\n--- MODALIDAD {mod.upper()} ---")
    print(f"Números sorteados: {nums}")
    print(f"-> Aciertos en Pool 12 ({len(hits_pool)}/6): {sorted(hits_pool)}")
    
    for t in tickets:
        t_nums = set(t['numeros'])
        matched = t_nums.intersection(set(nums))
        print(f"   * {t['nombre']}: {len(matched)} aciertos -> {sorted(list(matched))}")

print(f"\n--- POZO EXTRA ---")
print(f"Total números únicos sorteados (Tradicional + Segunda + Revancha): {sorted(list(pozo_extra_set))}")
hits_extra_pool = [n for n in sorted(list(pozo_extra_set)) if n in top_pool]
print(f"-> Aciertos en Pool 12 ({len(hits_extra_pool)} aciertos): {hits_extra_pool}")

for t in tickets:
    t_nums = set(t['numeros'])
    matched = t_nums.intersection(pozo_extra_set)
    print(f"   * {t['nombre']}: {len(matched)} aciertos en Pozo Extra -> {sorted(list(matched))}")
