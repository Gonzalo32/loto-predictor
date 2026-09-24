import json

pred_path = 'c:/Users/Administrador/Desktop/lot/proxima_prediccion.json'
with open(pred_path, 'r', encoding='utf-8') as f:
    pred = json.load(f)

top_pool = set(pred['top_numeros'])
tickets = pred['tickets']

print("=== PREDICCIÓN CALCULADA AYER (23/09/2026) ===")
print("Modelo:", pred['juego'])
print("Top Pool (12 números):", sorted(list(top_pool)))
for t in tickets:
    print(f"  {t['nombre']}: {t['numeros']}")

sorteo_3411 = {
    'sorteo': 3411, 'fecha': '2026-09-23',
    'Tradicional': [5, 7, 18, 25, 30, 41],
    'Segunda': [0, 6, 8, 14, 40, 44],
    'Revancha': [14, 15, 31, 34, 35, 41],
    'SiempreSale': [13, 24, 27, 32, 42, 44]
}

def evaluar(sorteo):
    print(f"\n==================================================")
    print(f"RESULTADOS Y EVALUACIÓN SORTEO #{sorteo['sorteo']} ({sorteo['fecha']})")
    print(f"==================================================")
    
    pozo_extra_set = set(sorteo['Tradicional'] + sorteo['Segunda'] + sorteo['Revancha'])
    
    for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
        nums = sorteo[mod]
        hits = [n for n in nums if n in top_pool]
        print(f"\nModalidad {mod}: {nums}")
        print(f"  -> Aciertos en Pool 12 ({len(hits)}/6): {hits}")
        for t in tickets:
            t_nums = set(t['numeros'])
            matched = t_nums.intersection(set(nums))
            print(f"     - {t['nombre']}: {len(matched)} aciertos -> {sorted(list(matched))}")

    print(f"\nPozo Extra (Números únicos en Tradicional, Segunda y Revancha): {sorted(list(pozo_extra_set))}")
    hits_extra = [n for n in sorted(list(pozo_extra_set)) if n in top_pool]
    print(f"  -> Aciertos en Pool 12 en Pozo Extra ({len(hits_extra)} aciertos): {hits_extra}")
    for t in tickets:
        t_nums = set(t['numeros'])
        matched = t_nums.intersection(pozo_extra_set)
        print(f"     - {t['nombre']}: {len(matched)} aciertos -> {sorted(list(matched))}")

evaluar(sorteo_3411)
