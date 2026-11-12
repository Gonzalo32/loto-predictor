import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
from collections import Counter, defaultdict
import numpy as np
from generador_cobertura_ml import load_data, compute_antagonist_pairs

draws = load_data()
sorteos = sorted(list(set(d['sorteo'] for d in draws)))

# Test on 300 draws
test_sorteos = sorteos[-300:]

def evaluate_resonance_levels(test_s_list, bonus_levels=[18.0, 6.0, 0.0]):
    results = {b: {'t4': 0, 't3': 0, 'g4': 0, 'g3': 0, 'dry': 0} for b in bonus_levels}
    
    # We can evaluate the predictive power of top 9 pool with different resonance bonuses
    for s_idx in range(len(test_s_list) - 1):
        s_target = test_s_list[s_idx + 1]
        s_prev = test_s_list[s_idx]
        
        hist_draws = [d for d in draws if d['sorteo'] <= s_prev]
        target_draws = [d for d in draws if d['sorteo'] == s_target]
        target_trad = [d for d in target_draws if d['modalidad'] == 'Tradicional']
        if not target_trad: continue
        target_nums = set(target_trad[0]['numbers'])
        
        # Base counts
        trad_hist = [d for d in hist_draws if d['modalidad'] == 'Tradicional']
        if len(trad_hist) < 30: continue
        
        c_60 = Counter(n for d in trad_hist[-60:] for n in d['numbers'])
        c_20 = Counter(n for d in trad_hist[-20:] for n in d['numbers'])
        
        # Last date multi-modal counts
        last_date = [d for d in hist_draws if d['sorteo'] == s_prev]
        res_counts = Counter(n for d in last_date for n in d['numbers'])
        
        for bon in bonus_levels:
            scores = defaultdict(float)
            for n in range(46):
                scores[n] = 1.5 * c_60[n] + 2.5 * c_20[n]
                if res_counts[n] >= 2:
                    scores[n] += bon * (res_counts[n] - 1)
                elif res_counts[n] == 1:
                    scores[n] += (bon / 4.5)
            
            top9 = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:9]
            top9_set = set(top9)
            
            # Form top wheel combo (take top 6)
            b1 = set(top9[:6])
            hits = len(b1 & target_nums)
            pool_hits = len(top9_set & target_nums)
            
            if hits >= 4: results[bon]['t4'] += 1
            if hits == 3: results[bon]['t3'] += 1
            if hits <= 1: results[bon]['dry'] += 1
            if pool_hits >= 4: results[bon]['g4'] += 1
            if pool_hits >= 3: results[bon]['g3'] += 1
            
    return results

res = evaluate_resonance_levels(test_sorteos, [18.0, 6.0, 0.0])
print("==================================================")
print("BENCHMARK COMPARATIVO DE BONUS DE RESONANCIA (300 SORTEOS):")
print("==================================================")
for bon, data in res.items():
    print(f"Bonus {bon:>4.1f} pts -> Boleto 1: 4+ hits: {data['t4']}, 3 hits: {data['t3']}, Sorteos Secos (<=1): {data['dry']} | Pool 9: 4+ hits: {data['g4']}")
print("==================================================")

