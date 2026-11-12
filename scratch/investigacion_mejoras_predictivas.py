import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from generador_cobertura_ml import load_data

draws = load_data()
sorteos = sorted(list(set(d['sorteo'] for d in draws)))

print("==================================================================")
print("INVESTIGACIÓN DE NUEVAS VÍAS DE MEJORA PREDICTIVA (659 FECHAS)")
print("==================================================================")

# 1. TEST DE DÍA DE LA SEMANA (DOMINGO VS MIÉRCOLES)
wed_draws = [d for d in draws if d['modalidad'] == 'Tradicional' and datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == 2]
sun_draws = [d for d in draws if d['modalidad'] == 'Tradicional' and datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == 6]

print(f"Total Sorteos Tradicional Miércoles : {len(wed_draws)}")
print(f"Total Sorteos Tradicional Domingo   : {len(sun_draws)}")

# Calculate frequency correlation between Sunday and Wednesday
freq_wed = Counter(n for d in wed_draws for n in d['numbers'])
freq_sun = Counter(n for d in sun_draws for n in d['numbers'])

vec_wed = np.array([freq_wed[n] for n in range(46)], dtype=float) / len(wed_draws)
vec_sun = np.array([freq_sun[n] for n in range(46)], dtype=float) / len(sun_draws)

corr_day = np.corrcoef(vec_wed, vec_sun)[0, 1]
print(f"Correlación de frecuencia entre Miércoles y Domingo: {corr_day:.4f}")

# Check top 5 balls on Wednesday vs Sunday
top5_wed = sorted(range(46), key=lambda x: vec_wed[x], reverse=True)[:5]
top5_sun = sorted(range(46), key=lambda x: vec_sun[x], reverse=True)[:5]
print(f"Top 5 históricas Miércoles: {top5_wed}")
print(f"Top 5 históricas Domingo  : {top5_sun}")

# 2. TEST DE MICRO-CLIMA DE FECHA (MISMA NOCHE, DIFERENTES MODALIDADES)
# Does the union of Tradicional + Segunda predict Revancha / Siempre Sale better than random?
same_night_repeats = []
for s in sorteos:
    d_s = [d for d in draws if d['sorteo'] == s]
    if len(d_s) >= 4:
        trad = set(d_s[0]['numbers'])
        seg = set(d_s[1]['numbers'])
        rev = set(d_s[2]['numbers'])
        ss = set(d_s[3]['numbers'])
        
        # How many balls repeat within the SAME night?
        # Under pure independence, E[Trad & Seg] = 6 * (6/46) = 0.7826 balls
        same_night_repeats.append(len(trad & seg))

avg_same_night = np.mean(same_night_repeats)
print(f"\nRepetición promedio entre Tradicional y Segunda en la MISMA noche: {avg_same_night:.3f} bolillas (Teórico azar: 0.783)")
print(f"Lift intra-noche: {avg_same_night / 0.7826:.3f}x")

# 3. TEST DE INTER-ARRIVAL TIME (POISSON MEMORY VS RENEWAL PROCESS)
# Is inter-arrival time memoryless (exponential) or Weibull/Gamma (clustering)?
delays = defaultdict(list)
last_seen = {}
for idx, d in enumerate(draws):
    if d['modalidad'] == 'Tradicional':
        for n in d['numbers']:
            if n in last_seen:
                delays[n].append(idx - last_seen[n])
            last_seen[n] = idx

all_delays = [gap for n in delays for gap in delays[n]]
mean_delay = np.mean(all_delays)
std_delay = np.std(all_delays)
cv = std_delay / mean_delay
print(f"\nCoeficiente de Variación (CV) de demoras en Tradicional: {cv:.3f}")
print("  (CV = 1.0 -> Proceso Poisson puro sin memoria)")
print(f"  (CV = {cv:.3f} -> {'Memoria / Agrupamiento detectable (CV != 1.0)' if abs(cv-1.0) > 0.05 else 'Cercano a Poisson'})")

print("==================================================================")

