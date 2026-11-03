import json
import sys

pred = json.load(open('c:/Users/Administrador/Desktop/lot/proxima_prediccion.json', encoding='utf-8'))
top = set(pred['top_numeros'])
t_a = set(pred['tickets'][0]['numeros'])
t_b = set(pred['tickets'][1]['numeros'])
t_c = set(pred['tickets'][2]['numeros'])

sorteos = {
    3420: {
        'fecha': '2026-10-25 (Domingo pasado)',
        'Tradicional': [0, 2, 3, 8, 19, 39],
        'Segunda': [8, 25, 30, 36, 43, 45],
        'Revancha': [5, 22, 23, 29, 41, 43],
        'SiempreSale': [0, 13, 22, 24, 40, 43]
    },
    3419: {
        'fecha': '2026-10-21 (Miércoles anterior)',
        'Tradicional': [4, 5, 32, 39, 40, 44],
        'Segunda': [4, 20, 21, 24, 42, 45],
        'Revancha': [1, 2, 11, 14, 28, 33],
        'SiempreSale': [13, 21, 39, 42, 43, 45]
    }
}

print(f"Predicción fecha: {pred['fecha_prediccion']}")
print(f"Top números (pool de {len(top)}): {sorted(list(top))}")
print(f"Ticket A: {sorted(list(t_a))}")
print(f"Ticket B: {sorted(list(t_b))}")
print(f"Ticket C: {sorted(list(t_c))}\n")

for s_num, s_data in sorteos.items():
    print("=" * 60)
    print(f"EVALUACIÓN CONTRA SORTEO #{s_num} - {s_data['fecha']}")
    print("=" * 60)
    for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
        nums = set(s_data[mod])
        pool_hits = sorted(list(top & nums))
        hits_a = sorted(list(t_a & nums))
        hits_b = sorted(list(t_b & nums))
        hits_c = sorted(list(t_c & nums))
        print(f"\n--- Modalidad: {mod} | Salió: {sorted(list(nums))} ---")
        print(f"Aciertos Pool ({len(top)}): {len(pool_hits)} -> {pool_hits}")
        print(f"Ticket A: {len(hits_a)} aciertos -> {hits_a}")
        print(f"Ticket B: {len(hits_b)} aciertos -> {hits_b}")
        print(f"Ticket C: {len(hits_c)} aciertos -> {hits_c}")
