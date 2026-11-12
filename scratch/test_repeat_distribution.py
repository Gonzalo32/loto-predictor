import os, sys
from collections import Counter
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)
from generador_cobertura_ml import load_data

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
sorteos = sorted(list(set(d['sorteo'] for d in draws)))

repeats_from_prev_date = []
repeats_from_prev_trad = []

for idx in range(1, len(sorteos)):
    prev_s = sorteos[idx-1]
    curr_s = sorteos[idx]
    
    prev_draws = [d for d in draws if d['sorteo'] == prev_s]
    curr_trad = [d for d in draws if d['sorteo'] == curr_s and d['modalidad'] == 'Tradicional']
    if not curr_trad: continue
    
    prev_all_nums = set(n for d in prev_draws for n in d['numbers'])
    prev_trad_nums = set(n for d in prev_draws if d['modalidad'] == 'Tradicional' for n in d['numbers'])
    curr_trad_nums = set(curr_trad[0]['numbers'])
    
    rep_date = len(curr_trad_nums & prev_all_nums)
    rep_trad = len(curr_trad_nums & prev_trad_nums)
    
    repeats_from_prev_date.append(rep_date)
    repeats_from_prev_trad.append(rep_trad)

print("Distribucion de repeticiones en Tradicional desde la FECHA PREVIA COMPLETA (24 bolillas):")
cnt_date = Counter(repeats_from_prev_date)
total = len(repeats_from_prev_date)
for k in range(7):
    print(f"  {k} numeros repetidos de la fecha previa: {cnt_date[k]} veces ({cnt_date[k]/total*100:.1f}%)")
print(f"Media de numeros que se repiten de la fecha previa: {np.mean(repeats_from_prev_date):.2f}")

print("\nDistribucion de repeticiones en Tradicional desde el Tradicional anterior (6 bolillas):")
cnt_trad = Counter(repeats_from_prev_trad)
for k in range(7):
    print(f"  {k} numeros repetidos del Tradicional previo: {cnt_trad[k]} veces ({cnt_trad[k]/total*100:.1f}%)")
print(f"Media de numeros que se repiten del Tradicional previo: {np.mean(repeats_from_prev_trad):.2f}")

