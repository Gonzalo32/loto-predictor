import json
import csv

with open('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv', 'r') as f:
    r = csv.reader(f)
    header = next(r)
    draws_data = {}
    for row in r:
        s = int(row[0])
        mod = row[2]
        nums = [int(x) for x in row[3:9]]
        if s not in draws_data:
            draws_data[s] = {'sorteo': s, 'fecha': row[1]}
        draws_data[s][mod] = nums

with open('c:/Users/Administrador/Desktop/lot/historial_predicciones.json', 'r') as f:
    hist = json.load(f)

eval_pairs = [
    (3416, '2026-10-09T08:46:27.510857'),
    (3417, '2026-10-13T08:44:00.285588'),
    (3418, '2026-10-15T11:05:26.757225')
]

for s_num, pred_dt in eval_pairs:
    s_info = draws_data.get(s_num)
    pred_entry = next((p for p in hist if p.get('fecha_prediccion') == pred_dt), None)
    print("="*60)
    print(f"SORTEO #{s_num} ({s_info['fecha']})")
    pozo_extra = set(s_info['Tradicional'] + s_info['Segunda'] + s_info['Revancha'])
    print("  Tradicional:", s_info['Tradicional'])
    print("  Segunda:    ", s_info['Segunda'])
    print("  Revancha:   ", s_info['Revancha'])
    print("  SiempreSale:", s_info['SiempreSale'])
    print("  Pozo Extra (18):", sorted(list(pozo_extra)))
    
    if pred_entry:
        top_pool = pred_entry.get('top_numeros', [])
        hits_pool_pe = len(set(top_pool).intersection(pozo_extra))
        print(f"  Top Pool ({len(top_pool)}): {top_pool} -> {hits_pool_pe} en Pozo Extra")
        for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
            h_mod = len(set(top_pool).intersection(set(s_info[mod])))
            print(f"    Hits Pool en {mod:11s}: {h_mod}/6")
        for t in pred_entry['tickets']:
            t_nums = set(t['numeros'])
            h_trad = len(t_nums.intersection(set(s_info['Tradicional'])))
            h_seg = len(t_nums.intersection(set(s_info['Segunda'])))
            h_rev = len(t_nums.intersection(set(s_info['Revancha'])))
            h_ss = len(t_nums.intersection(set(s_info['SiempreSale'])))
            h_pe = len(t_nums.intersection(pozo_extra))
            print(f"    Ticket: {t['nombre']} {sorted(list(t_nums))}")
            print(f"      Hits -> Trad:{h_trad} | Seg:{h_seg} | Rev:{h_rev} | SS:{h_ss} | PozoExtra:{h_pe}")
