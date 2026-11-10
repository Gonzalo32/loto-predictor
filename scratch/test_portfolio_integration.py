"""
Test de integracion para generador_cobertura_ml con todas las mejoras:
- Filtro 4 Decenas
- Bloqueo Antagonico
- Modelado Dia de Semana
- Portfolio Especializado de 3 Boletos
"""
import os, sys, itertools
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

WORKSPACE_DIR = r'c:\Users\Administrador\Desktop\lot'
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, compute_exponential_cooccurrence_scores,
    compute_delta_delay_momentum_scores, compute_attention_mlp_scores,
    estimar_probabilidad_exito_conjunto, prob_hits_poisson_binomial_flat
)
from cobertura_combinatoria import select_optimal_wheel_triplet

def compute_antagonist_pairs(train_draws, min_expected=7.0, max_obs=2):
    T_hist = len(train_draws)
    if T_hist < 50: return set()
    freq_hist = Counter(n for d in train_draws for n in d['numbers'])
    pair_cnt = Counter(tuple(sorted([u, v])) for d in train_draws for u, v in itertools.combinations(d['numbers'], 2))
    antagonist_pairs = set()
    for u, v in itertools.combinations(range(46), 2):
        p_exp = (freq_hist[u] / T_hist) * (freq_hist[v] / T_hist) * T_hist
        if p_exp >= min_expected and pair_cnt[tuple(sorted([u, v]))] <= max_obs:
            antagonist_pairs.add(tuple(sorted([u, v])))
    return antagonist_pairs

def es_ticket_valido_mejorado(ticket, recent_numbers=None, saturated_decades=None, antagonist_pairs=None, min_distinct_decades=4):
    ticket = sorted(ticket)
    # 1. Consecutividad: max 2 parejas
    if sum(1 for i in range(1, len(ticket)) if ticket[i] - ticket[i-1] <= 1) > 2:
        return False
    # 2. Spread: entre 15 y 45
    spread = ticket[5] - ticket[0]
    if spread < 15 or spread > 45:
        return False
    # 3. Decenas: max 3 por decena y max 2 en la del 40
    decenas = defaultdict(int)
    for num in ticket: decenas[num // 10] += 1
    if any(c > 3 for c in decenas.values()) or decenas[4] > 2:
        return False
    # 3b. Dispersion en decenas
    if len(decenas) < min_distinct_decades:
        return False
    # 4. Paridad: entre 1 y 5 pares
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares == 0 or pares == 6:
        return False
    # 5. Hot/cold: 1 a 5 calientes
    if recent_numbers is not None:
        rec_cnt = sum(1 for n in ticket if n in recent_numbers)
        if rec_cnt < 1 or rec_cnt > 5:
            return False
    # 6. Decada saturada
    if saturated_decades:
        for dec in saturated_decades:
            if decenas[dec] > 2:
                return False
    # 7. Antagonicos
    if antagonist_pairs:
        for u, v in itertools.combinations(ticket, 2):
            if tuple(sorted([u, v])) in antagonist_pairs:
                return False
    return True

draws = load_data()
antag_pairs = compute_antagonist_pairs(draws)
print(f"Total pares antagonicos detectados: {len(antag_pairs)}")

# Proximo sorteo
last_draw = draws[-1]
last_weekday = datetime.strptime(last_draw['fecha'], '%Y-%m-%d').weekday()
target_weekday = 2 if last_weekday == 6 else 6
weekday_name = "Miercoles" if target_weekday == 2 else "Domingo"
print(f"Ultimo sorteo: #{last_draw['sorteo']} ({last_draw['fecha']}). Proximo sorteo: {weekday_name}")

