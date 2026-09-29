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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generador_cobertura_ml import (
    load_data, compute_attention_mlp_scores,
    compute_cooccurrence_matrix, select_tiered_clique_tickets, es_ticket_valido
)
from cobertura_combinatoria import select_wheel_coverage_tickets

def ejecutar_backtest_completo():
    csv_path = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
    draws = load_data(csv_path)
    if not draws:
        print(f"[ERROR] No se pudo leer {csv_path}")
        return

    # Evaluaremos sobre los últimos 50 sorteos históricos reales de la modalidad 'Tradicional'
    tradicional_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
    eval_draws = tradicional_draws[-50:]
    
    print("="*85)
    print("                BACKTESTING COMPLETO DE ALTO RENDIMIENTO (WALK-FORWARD)")
    print(f" Dataset: {len(draws)} registros históricos | Sorteos evaluados: {len(eval_draws)} (Últimos 50 sorteos)")
    print(f" Rango de Sorteos Evaluados: Sorteo #{eval_draws[0]['sorteo']} al Sorteo #{eval_draws[-1]['sorteo']}")
    print("="*85)

    # Acumuladores de Métricas para el Modelo Ensamble (Pool 12 - Configuración Producción)
    model_pool_hits = []
    model_ticket_max_hits = []
    model_3plus = 0
    model_4plus = 0
    model_5plus = 0
    model_6plus = 0

    # Acumuladores para Baseline Aleatorio (Azar Puro - 3 boletas de 6 números)
    random_ticket_max_hits = []
    random_3plus = 0
    random_4plus = 0
    random_5plus = 0

    # Acumuladores para Baseline de Frecuencia Simple (Números más salidores)
    freq_ticket_max_hits = []
    freq_3plus = 0
    freq_4plus = 0

    start_time = time.time()
    np.random.seed(42)

    for i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        target_nums = set(target_draw['numbers'])
        
        # Train data: solo datos HISTÓRICOS estrictamente anteriores al sorteo a predecir
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo or (d['sorteo'] == target_sorteo and d['modalidad'] != 'Tradicional')]
        
        if len(train_draws) < 100:
            continue

        # 1. FRECUENCIA RECIENTE (Últimos 60 sorteos)
        counts = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']:
                counts[n] += 1
                
        freq_scores = np.zeros(46)
        for n in range(46):
            freq_scores[n] = counts[n]

        # 2. RECENCIA PONDERADA EXPONENCIALMENTE
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']:
                recency_scores[n] += weight

        # 3. SCORE DE ATENCIÓN DE TRANSICIÓN MATRICIAL / MLP
        attn_scores = compute_attention_mlp_scores(train_draws)

        # ENSAMBLE DE MODELO: 30% Frecuencia + 40% Recencia + 30% Atención
        final_scores = freq_scores * 0.3 + recency_scores * 0.4 + attn_scores * 0.3
        borda_ranking = list(np.argsort(final_scores)[::-1])

        # POOL DE PRODUCCIÓN DE 12 NÚMEROS
        top12 = borda_ranking[:12]
        pool_hits = len(set(top12).intersection(target_nums))
        model_pool_hits.append(pool_hits)

        # GENERACIÓN DE TICKETS OPTIMIZADOS (HYPERGRAPH COVERAGE)
        cooccur_matrix = compute_cooccurrence_matrix(train_draws)
        combos12 = list(itertools.combinations(sorted(top12), 6))
        viables12 = [c for c in combos12 if es_ticket_valido(c)]
        if not viables12:
            viables12 = combos12[:50]
        
        model_tickets = select_wheel_coverage_tickets(viables12, final_scores, n_tickets=3, target_subsize=4)
        
        # Evaluar aciertos máximos por boleta en el Modelo
        max_h_model = max(len(set(t).intersection(target_nums)) for t in model_tickets)
        model_ticket_max_hits.append(max_h_model)

        if max_h_model >= 3: model_3plus += 1
        if max_h_model >= 4: model_4plus += 1
        if max_h_model >= 5: model_5plus += 1
        if max_h_model == 6: model_6plus += 1

        # BASELINE 1: AZAR PURO (3 boletas de 6 números tomados al azar entre 0 y 45)
        rnd_tickets = [set(np.random.choice(46, 6, replace=False)) for _ in range(3)]
        max_h_rnd = max(len(t.intersection(target_nums)) for t in rnd_tickets)
        random_ticket_max_hits.append(max_h_rnd)
        if max_h_rnd >= 3: random_3plus += 1
        if max_h_rnd >= 4: random_4plus += 1
        if max_h_rnd >= 5: random_5plus += 1

        # BASELINE 2: FRECUENCIA SIMPLE (Top 6 números más salidores históricos en 3 variantes)
        top_freq = list(np.argsort(freq_scores)[::-1])
        freq_tickets = [top_freq[:6], top_freq[1:7], top_freq[2:8]]
        max_h_freq = max(len(set(t).intersection(target_nums)) for t in freq_tickets)
        freq_ticket_max_hits.append(max_h_freq)
        if max_h_freq >= 3: freq_3plus += 1
        if max_h_freq >= 4: freq_4plus += 1

    elapsed = time.time() - start_time
    total_eval = len(model_ticket_max_hits)

    print("\n" + "="*85)
    print("                             RESULTADOS DEL BACKTEST")
    print("="*85)
    print(f" Sorteos Evaluados Validamente : {total_eval}")
    print(f" Tiempo Total de Cálculo       : {elapsed:.2f} segundos")
    print("-" * 85)

    print("\n 1. RENDIMIENTO DEL POOL DE 12 NÚMEROS (MODELO ENSAMBLE)")
    print(f"    - Promedio de Aciertos en Pool 12  : {np.mean(model_pool_hits):.2f} / 6 números ganadores")
    print(f"    - Tasa de Retención de Aciertos    : {(np.mean(model_pool_hits)/6)*100:.1f}% de los números del sorteo")
    print(f"    - % Sorteos con 3+ aciertos en Pool: {(sum(1 for h in model_pool_hits if h >= 3)/total_eval)*100:.1f}%")
    print(f"    - % Sorteos con 4+ aciertos en Pool: {(sum(1 for h in model_pool_hits if h >= 4)/total_eval)*100:.1f}%")

    print("\n 2. COMPARATIVA DE RENDIMIENTO POR BOLETAS DE 6 NÚMEROS (3 TICKETS POR SORTEO)")
    print(" " + "-"*81)
    print("  Métrica de Acierto              | Modelo ML Ensamble | Frecuencia Simple | Azar / Aleatorio")
    print(" " + "-"*81)
    print(f"  Promedio Aciertos Máx / Sorteo  |     {np.mean(model_ticket_max_hits):.2f}         |       {np.mean(freq_ticket_max_hits):.2f}       |      {np.mean(random_ticket_max_hits):.2f}")
    print(f"  % Sorteos con >= 3 Aciertos     |     {(model_3plus/total_eval)*100:5.1f}%        |       {(freq_3plus/total_eval)*100:5.1f}%      |      {(random_3plus/total_eval)*100:5.1f}%")
    print(f"  % Sorteos con >= 4 Aciertos     |     {(model_4plus/total_eval)*100:5.1f}%        |       {(freq_4plus/total_eval)*100:5.1f}%      |      {(random_4plus/total_eval)*100:5.1f}%")
    print(f"  % Sorteos con >= 5 Aciertos     |     {(model_5plus/total_eval)*100:5.1f}%        |       {(0.0):5.1f}%      |      {(random_5plus/total_eval)*100:5.1f}%")
    print(" " + "-"*81)

    print("\n 3. DISTRIBUCIÓN DE ACIERTOS MÁXIMOS POR SORTEO (MODELO ML)")
    hit_counts = Counter(model_ticket_max_hits)
    for h in range(7):
        count = hit_counts[h]
        pct = (count / total_eval) * 100
        bar = "#" * int(pct // 2)
        print(f"    - {h} aciertos: {count:2d} sorteos ({pct:5.1f}%) {bar}")

    # Guardar métricas de backtest en JSON
    results_summary = {
        'fecha_ejecucion': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_sorteos_evaluados': total_eval,
        'promedio_aciertos_pool12': float(np.mean(model_pool_hits)),
        'promedio_aciertos_ticket_modelo': float(np.mean(model_ticket_max_hits)),
        'promedio_aciertos_ticket_azar': float(np.mean(random_ticket_max_hits)),
        'pct_3_mas_aciertos_modelo': float((model_3plus/total_eval)*100),
        'pct_4_mas_aciertos_modelo': float((model_4plus/total_eval)*100),
        'pct_5_mas_aciertos_modelo': float((model_5plus/total_eval)*100),
        'distribucion_aciertos': dict(hit_counts)
    }

    out_json = 'c:/Users/Administrador/Desktop/lot/resultado_backtest.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[OK] Resultados del backtest guardados en '{out_json}'.\n")

if __name__ == '__main__':
    ejecutar_backtest_completo()
