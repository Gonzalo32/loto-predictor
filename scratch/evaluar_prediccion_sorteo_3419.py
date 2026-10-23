import json
import os

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

# Resultados oficiales Sorteo 3419 (21/10/2026)
sorteo_3419 = {
    'sorteo': 3419,
    'fecha': '2026-10-21',
    'Tradicional': [4, 5, 32, 39, 40, 44],
    'Segunda': [4, 20, 21, 24, 42, 45],
    'Revancha': [1, 2, 11, 14, 28, 33],
    'SiempreSale': [13, 21, 39, 42, 43, 45]
}

pozo_extra_set = set(sorteo_3419['Tradicional'] + sorteo_3419['Segunda'] + sorteo_3419['Revancha'])

print(f"EVALUACIÓN DETALLADA - SORTEO #{sorteo_3419['sorteo']} ({sorteo_3419['fecha']})")
print("="*60)

for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
    nums = sorteo_3419[mod]
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

