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
os.environ["LIGHTGBM_VERBOSE"] = "-1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generador_cobertura_ml import (
    load_data, compute_adaptive_pool,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    es_ticket_valido
)

def run_full_backtest():
    csv_path = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
    draws = load_data(csv_path)
    if not draws:
        print("[ERROR] No se pudo leer datos")
        return

    # Evaluaremos sobre los últimos 50 sorteos secuenciales de la modalidad Tradicional
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]
    
    print("="*85)
    print("          BACKTESTING GLOBAL EXHAUSTIVO (SIMULACIÓN REAL WALK-FORWARD)")
    print(f" Evaluando {len(eval_draws)} sorteos pasados (Sorteos #{eval_draws[0]['sorteo']} al #{eval_draws[-1]['sorteo']})")
    print("="*85)

    # Acumuladores del Nuevo Modelo Optimizado (Pool 14 + PCR + DDM + Core-Surge Tickets)
    pool_14_hits = []
    ticket_max_hits = []
    ticket_3plus = 0
    ticket_4plus = 0
    ticket_5plus = 0

    # Acumuladores por Modalidad
    hits_tradicional = []
    hits_segunda = []
    hits_revancha = []
    hits_siempresale = []
    hits_pozo_extra = []

    # Acumuladores Baseline Azar Puro (3 boletas de 6 números)
    random_max_hits = []
    random_3plus = 0
    random_4plus = 0

    # Acumuladores Baseline Frecuencia
    freq_max_hits = []
    freq_3plus = 0

    start_time = time.time()
    np.random.seed(42)

    for i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        
        # Obtener los resultados reales del sorteo objetivo en TODAS sus modalidades
        draw_group = [d for d in draws if d['sorteo'] == target_sorteo]
        mod_dict = {d['modalidad']: set(d['numbers']) for d in draw_group}
        
        target_trad = mod_dict.get('Tradicional', set())
        target_seg = mod_dict.get('Segunda', set())
        target_rev = mod_dict.get('Revancha', set())
        target_ss = mod_dict.get('SiempreSale', set())
        pozo_extra_set = target_trad.union(target_seg).union(target_rev)

        # Train data: estrictamente datos ANTERIORES al sorteo objetivo
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        if len(train_draws) < 100:
            continue

        # 1. Puntuaciones Frecuencia y Recencia
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1
                
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency_scores[n] += weight

        # 2. PCR y DDM
        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)

        # 3. Puntuación Ensamble Borda
        borda = defaultdict(float)
        models_to_rank = [
            (counts_60, 1.5),
            (recency_scores, 2.0),
            (pcr_scores, 1.4),
            (ddm_scores, 1.0)
        ]
        for p_model, weight in models_to_rank:
            if isinstance(p_model, Counter):
                ranking_m = sorted(range(46), key=lambda x: p_model[x], reverse=True)
            else:
                ranking_m = sorted(range(46), key=lambda x: p_model[x], reverse=True)
            for rank, n in enumerate(ranking_m):
                borda[n] += weight * (45 - rank)

        borda_ranking = [n for n, score in sorted(borda.items(), key=lambda x: x[1], reverse=True)]

        # Candidate Pool Adaptativo 14
        top_pool_14 = set(borda_ranking[:14])
        p14_hits = len(top_pool_14.intersection(target_trad))
        pool_14_hits.append(p14_hits)

        # Generación de 3 Tickets Híbridos Core-Surge
        core_pool = set(borda_ranking[:8])
        surge_pool = top_pool_14 - core_pool

        combos_all = list(itertools.combinations(sorted(top_pool_14), 6))
        viables_all = [c for c in combos_all if es_ticket_valido(c)] or combos_all[:100]

        # Ticket A (Core Spike)
        viables_core = [c for c in viables_all if sum(1 for n in c if n in core_pool) >= 4] or viables_all
        ticket_A = max(viables_core, key=lambda b: sum(borda[n] for n in b) + 200.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)))

        # Ticket B (Affinity Clique - distinto de A)
        viables_B = [c for c in viables_all if set(c) != set(ticket_A)] or viables_all
        ticket_B = max(viables_B, key=lambda b: sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda[n] for n in b))

        # Ticket C (Hypergraph Surge Cover - distinto de A y B)
        borda_C = borda.copy()
        for n in set(ticket_A).union(set(ticket_B)): borda_C[n] *= 0.5
        viables_C = [c for c in viables_all if set(c) != set(ticket_A) and set(c) != set(ticket_B)] or viables_all
        ticket_C = max(viables_C, key=lambda b: sum(borda_C[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))

        model_tickets = [set(ticket_A), set(ticket_B), set(ticket_C)]

        # Evaluar aciertos en Tradicional
        max_h_trad = max(len(t.intersection(target_trad)) for t in model_tickets)
        ticket_max_hits.append(max_h_trad)
        if max_h_trad >= 3: ticket_3plus += 1
        if max_h_trad >= 4: ticket_4plus += 1
        if max_h_trad >= 5: ticket_5plus += 1

        # Evaluar aciertos en otras modalidades
        if target_trad: hits_tradicional.append(max_h_trad)
        if target_seg: hits_segunda.append(max(len(t.intersection(target_seg)) for t in model_tickets))
        if target_rev: hits_revancha.append(max(len(t.intersection(target_rev)) for t in model_tickets))
        if target_ss: hits_siempresale.append(max(len(t.intersection(target_ss)) for t in model_tickets))
        if pozo_extra_set: hits_pozo_extra.append(max(len(t.intersection(pozo_extra_set)) for t in model_tickets))

        # Baseline Azar Puro
        rnd_tickets = [set(np.random.choice(46, 6, replace=False)) for _ in range(3)]
        max_h_rnd = max(len(t.intersection(target_trad)) for t in rnd_tickets)
        random_max_hits.append(max_h_rnd)
        if max_h_rnd >= 3: random_3plus += 1
        if max_h_rnd >= 4: random_4plus += 1

        # Baseline Frecuencia
        top_freq = borda_ranking[:8]
        freq_tickets = [set(top_freq[:6]), set(top_freq[1:7]), set(top_freq[2:8])]
        max_h_freq = max(len(t.intersection(target_trad)) for t in freq_tickets)
        freq_max_hits.append(max_h_freq)
        if max_h_freq >= 3: freq_3plus += 1

    elapsed = time.time() - start_time
    total = len(ticket_max_hits)

    print("\n" + "="*85)
    print("                        RESUMEN RESULTADOS BACKTESTING")
    print("="*85)
    print(f" Sorteos Evaluados: {total} | Tiempo Total: {elapsed:.2f} segundos")
    print("-" * 85)

    print("\n 1. RENDIMIENTO DEL POOL CANDIDATO DE 14 NÚMEROS")
    print(f"    - Promedio Aciertos en Pool 14    : {np.mean(pool_14_hits):.2f} / 6 números sorteados")
    print(f"    - Tasa de Retención del Pool      : {(np.mean(pool_14_hits)/6)*100:.1f}% de la combinación ganadora")
    print(f"    - % Sorteos con 3+ aciertos Pool  : {(sum(1 for h in pool_14_hits if h >= 3)/total)*100:.1f}%")
    print(f"    - % Sorteos con 4+ aciertos Pool  : {(sum(1 for h in pool_14_hits if h >= 4)/total)*100:.1f}%")

    print("\n 2. COMPARATIVA DE RENDIMIENTO POR BOLETAS (3 TICKETS DE 6 NÚMEROS POR SORTEO)")
    print(" " + "-"*81)
    print("  Métrica de Rendimiento           | Modelo Nuevo Optimizado | Frecuencia | Azar / Aleatorio")
    print(" " + "-"*81)
    print(f"  Promedio Aciertos Máx / Sorteo   |          {np.mean(ticket_max_hits):.2f}          |    {np.mean(freq_max_hits):.2f}    |       {np.mean(random_max_hits):.2f}")
    print(f"  % Sorteos con >= 3 Aciertos      |          {(ticket_3plus/total)*100:5.1f}%         |    {(freq_3plus/total)*100:5.1f}%   |       {(random_3plus/total)*100:5.1f}%")
    print(f"  % Sorteos con >= 4 Aciertos      |          {(ticket_4plus/total)*100:5.1f}%         |    {(0.0):5.1f}%   |       {(random_4plus/total)*100:5.1f}%")
    print(" " + "-"*81)

    print("\n 3. RENDIMIENTO DEL MODELO EXPANDIDO A OTRAS MODALIDADES Y POZO EXTRA")
    print(f"    - Promedio Aciertos Máx en Tradicional   : {np.mean(hits_tradicional):.2f} nros")
    print(f"    - Promedio Aciertos Máx en La Segunda    : {np.mean(hits_segunda):.2f} nros")
    print(f"    - Promedio Aciertos Máx en Revancha      : {np.mean(hits_revancha):.2f} nros")
    print(f"    - Promedio Aciertos Máx en Siempre Sale  : {np.mean(hits_siempresale):.2f} nros")
    print(f"    - Promedio Aciertos Máx en POZO EXTRA    : {np.mean(hits_pozo_extra):.2f} nros (de 15 nros únicos)")
    print(f"    - % Sorteos con 3+ aciertos en Pozo Extra: {(sum(1 for h in hits_pozo_extra if h >= 3)/len(hits_pozo_extra))*100:.1f}%")
    print("="*85)

    # Guardar métricas en JSON
    results = {
        'total_sorteos': total,
        'tiempo_calculo': elapsed,
        'pool_14_promedio_aciertos': float(np.mean(pool_14_hits)),
        'modelo_promedio_aciertos_ticket': float(np.mean(ticket_max_hits)),
        'azar_promedio_aciertos_ticket': float(np.mean(random_max_hits)),
        'pct_3_plus_modelo': float((ticket_3plus/total)*100),
        'pct_4_plus_modelo': float((ticket_4plus/total)*100),
        'pozo_extra_promedio_aciertos': float(np.mean(hits_pozo_extra)),
        'pct_3_plus_pozo_extra': float((sum(1 for h in hits_pozo_extra if h >= 3)/len(hits_pozo_extra))*100)
    }

    with open('resultado_backtest.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print("\n[OK] Resultados guardados en resultado_backtest.json")

if __name__ == '__main__':
    run_full_backtest()
