import csv
import json
import numpy as np
import itertools
import os
import sys
from collections import defaultdict, Counter
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generador_cobertura_ml import load_data, es_ticket_valido, compute_attention_mlp_scores

def run_core_surge_backtest():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    if not draws:
        return

    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]
    
    print("="*85)
    print(" TESTING ESTRATEGIA HÍBRIDA CORE-SURGE (CORE POOL TOP 8 + SURGE POOL TOP 14)")
    print("="*85)

    base_max_hits = []
    base_3plus = 0
    base_4plus = 0

    core_surge_max_hits = []
    cs_3plus = 0
    cs_4plus = 0

    for i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        target_nums = set(target_draw['numbers'])
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo or (d['sorteo'] == target_sorteo and d['modalidad'] != 'Tradicional')]
        if len(train_draws) < 100:
            continue

        # Scores
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1
                
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency_scores[n] += weight

        cooccur_exp = np.zeros((46, 46))
        for t_idx, d in enumerate(train_draws[-40:]):
            w_t = np.exp((t_idx - 40) / 15.0)
            for u, v in itertools.combinations(d['numbers'], 2):
                cooccur_exp[u, v] += w_t
                cooccur_exp[v, u] += w_t

        cooccur_bonus = np.array([np.sum(cooccur_exp[n]) for n in range(46)])

        scores = np.zeros(46)
        for n in range(46):
            scores[n] = (
                counts_60[n] * 0.35 + 
                recency_scores[n] * 0.40 + 
                (cooccur_bonus[n] / (np.max(cooccur_bonus) + 1e-5)) * 0.25
            )

        ranking = list(np.argsort(scores)[::-1])

        # BASELINE (Pool 12 uniforme)
        pool_12 = ranking[:12]
        combos_12 = list(itertools.combinations(pool_12, 6))
        viables_12 = [c for c in combos_12 if es_ticket_valido(c)] or combos_12[:3]
        # Pick top 3 by score sum
        viables_12.sort(key=lambda b: sum(scores[n] for n in b), reverse=True)
        t_base = viables_12[:3]
        max_base = max(len(set(t).intersection(target_nums)) for t in t_base)
        base_max_hits.append(max_base)
        if max_base >= 3: base_3plus += 1
        if max_base >= 4: base_4plus += 1

        # CORE-SURGE HÍBRIDO:
        # Core Pool: Top 8
        core_pool = ranking[:8]
        # Surge Pool: Top 9 a 14
        surge_pool = ranking[8:14]
        full_pool = ranking[:14]

        # Ticket A (Core Spike): Mejor ticket formado exclusivamente por el Core Pool (top 8)
        combos_core = list(itertools.combinations(core_pool, 6))
        viables_core = [c for c in combos_core if es_ticket_valido(c)] or combos_core
        ticket_A = max(viables_core, key=lambda b: sum(scores[n] for n in b))

        # Ticket B (Pair Co-occurrence Clique): Mejor ticket de sinergia de pares en el Top 10
        combos_10 = list(itertools.combinations(ranking[:10], 6))
        viables_10 = [c for c in combos_10 if es_ticket_valido(c)] or combos_10
        ticket_B = max(viables_10, key=lambda b: sum(cooccur_exp[u, v] for u, v in itertools.combinations(b, 2)) * 10.0 + sum(scores[n] for n in b))

        # Ticket C (Hypergraph Surge Cover): Mejor ticket que cubre elementos del Surge Pool con alta puntuación
        combos_full = list(itertools.combinations(full_pool, 6))
        # Exigir al menos 2 números del surge pool
        combos_surge = [c for c in combos_full if sum(1 for n in c if n in surge_pool) >= 2 and es_ticket_valido(c)]
        if not combos_surge: combos_surge = combos_full
        ticket_C = max(combos_surge, key=lambda b: sum(scores[n] for n in b))

        cs_tickets = [ticket_A, ticket_B, ticket_C]
        max_cs = max(len(set(t).intersection(target_nums)) for t in cs_tickets)
        core_surge_max_hits.append(max_cs)
        if max_cs >= 3: cs_3plus += 1
        if max_cs >= 4: cs_4plus += 1

    total = len(eval_draws)
    print("\n" + "="*85)
    print("                             RESULTADOS EVALUACIÓN")
    print("="*85)
    print(f" Métrica Evaluada                    | Baseline (Pool 12 Uniforme) | Core-Surge Hybrid (Top 8 Core + Top 14 Surge)")
    print(" " + "-"*81)
    print(f" Promedio Acierto Máx por Boleta     |          {np.mean(base_max_hits):.2f}            |                  {np.mean(core_surge_max_hits):.2f}")
    print(f" % Sorteos con >= 3 Aciertos Boleta  |          {(base_3plus/total)*100:5.1f}%           |                  {(cs_3plus/total)*100:5.1f}%")
    print(f" % Sorteos con >= 4 Aciertos Boleta  |          {(base_4plus/total)*100:5.1f}%           |                  {(cs_4plus/total)*100:5.1f}%")
    print(" " + "-"*81)

if __name__ == '__main__':
    run_core_surge_backtest()
