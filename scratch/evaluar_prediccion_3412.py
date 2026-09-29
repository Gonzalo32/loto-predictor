import json

pred_path = 'c:/Users/Administrador/Desktop/lot/proxima_prediccion.json'
with open(pred_path, 'r', encoding='utf-8') as f:
    pred = json.load(f)

top_pool = sorted(pred['top_numeros'])
tickets = pred['tickets']

print("==================================================")
print("PREDICCIÓN GENERADA EL:", pred.get('fecha_prediccion', 'N/A'))
print("Modelo:", pred.get('juego', 'Quini 6'))
print("Top Pool (12 números):", top_pool)
print("Tickets jugados:")
for t in tickets:
    print(f"  - {t['nombre']}: {t['numeros']}")
print("==================================================")

sorteo_3412 = {
    'sorteo': 3412,
    'fecha': '2026-09-27 (Domingo)',
    'Tradicional': [3, 4, 19, 21, 31, 36],
    'Segunda': [3, 8, 11, 12, 14, 21],
    'Revancha': [8, 25, 29, 34, 42, 43],
    'SiempreSale': [1, 7, 11, 31, 32, 45]
}

print(f"\nRESULTADOS SORTEO #{sorteo_3412['sorteo']} ({sorteo_3412['fecha']})")
print("-" * 50)

pozo_extra_set = set(sorteo_3412['Tradicional'] + sorteo_3412['Segunda'] + sorteo_3412['Revancha'])

for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
    nums = sorteo_3412[mod]
    hits_pool = [n for n in nums if n in top_pool]
    print(f"\nModalidad {mod}: {nums}")
    print(f"  -> Aciertos en Pool 12 ({len(hits_pool)}/6): {hits_pool}")
    for t in tickets:
        t_set = set(t['numeros'])
        matched = sorted(list(t_set.intersection(set(nums))))
        print(f"     - {t['nombre']}: {len(matched)} aciertos -> {matched}")

print("\n" + "=" * 50)
print(f"Pozo Extra (Números únicos en Tradicional, Segunda y Revancha): {sorted(list(pozo_extra_set))}")
hits_extra = [n for n in sorted(list(pozo_extra_set)) if n in top_pool]
print(f"  -> Aciertos en Pool 12 en Pozo Extra ({len(hits_extra)} aciertos): {hits_extra}")
for t in tickets:
    t_set = set(t['numeros'])
    matched = sorted(list(t_set.intersection(pozo_extra_set)))
    print(f"     - {t['nombre']}: {len(matched)} aciertos -> {matched}")
