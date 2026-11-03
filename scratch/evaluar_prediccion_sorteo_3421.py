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

# Resultados oficiales Sorteo 3421 (28/10/2026 - Miércoles de ayer)
sorteo_3421 = {
    'sorteo': 3421,
    'fecha': '2026-10-28',
    'Tradicional': [6, 25, 34, 36, 38, 45],
    'Segunda': [9, 19, 26, 29, 33, 44],
    'Revancha': [22, 26, 35, 42, 43, 45],
    'SiempreSale': [20, 22, 26, 35, 36, 37]
}

pozo_extra_set = set(sorteo_3421['Tradicional'] + sorteo_3421['Segunda'] + sorteo_3421['Revancha'])

print(f"EVALUACIÓN DETALLADA - SORTEO #{sorteo_3421['sorteo']} ({sorteo_3421['fecha']})")
print("="*60)

for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
    nums = sorteo_3421[mod]
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
