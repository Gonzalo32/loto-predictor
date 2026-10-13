import json
import pandas as pd

def main():
    with open('proxima_prediccion.json', 'r', encoding='utf-8') as f:
        pred = json.load(f)

    top_pool = sorted(pred['top_numeros'])
    tickets = pred['tickets']

    df = pd.read_csv('historico_quini_completo.csv')
    sorteo_3416 = df[df['Sorteo'] == 3416]

    print("=== PREDICCIÓN GENERADA EL VIERNES 09/10/2026 ===")
    print(f"Fecha predicción: {pred.get('fecha_prediccion')}")
    print(f"Top Pool 14: {top_pool}")
    print("Boletos jugados:")
    for t in tickets:
        print(f"  - {t['nombre']}: {t['numeros']}")

    print("\n=== RESULTADOS SORTEO #3416 (DOMINGO 11/10/2026) ===")
    modalidades = {}
    for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
        row = sorteo_3416[sorteo_3416['Modalidad'] == mod].iloc[0]
        nums = sorted([int(row[f'B{i}']) for i in range(1, 7)])
        modalidades[mod] = nums
        hits_pool = [n for n in nums if n in top_pool]
        print(f"\nModalidad {mod}: {nums}")
        print(f"  -> Aciertos en Pool 14 ({len(hits_pool)}/6): {hits_pool}")
        for t in tickets:
            t_nums = set(t['numeros'])
            matched = sorted(list(t_nums.intersection(set(nums))))
            print(f"     * {t['nombre']}: {len(matched)} aciertos -> {matched}")

    pozo_extra = sorted(list(set(modalidades['Tradicional'] + modalidades['Segunda'] + modalidades['Revancha'])))
    print(f"\nPozo Extra ({len(pozo_extra)} números únicos): {pozo_extra}")
    hits_extra_pool = [n for n in pozo_extra if n in top_pool]
    print(f"  -> Aciertos en Pool 14 en Pozo Extra ({len(hits_extra_pool)}/{len(pozo_extra)}): {hits_extra_pool}")
    for t in tickets:
        t_nums = set(t['numeros'])
        matched = sorted(list(t_nums.intersection(set(pozo_extra))))
        print(f"     * {t['nombre']}: {len(matched)} aciertos en Pozo Extra -> {matched}")

if __name__ == '__main__':
    main()
