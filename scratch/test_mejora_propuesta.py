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

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generador_cobertura_ml import load_data, es_ticket_valido
from cobertura_combinatoria import select_wheel_coverage_tickets

def run_improvement_backtest():
    csv_path = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
    draws = load_data(csv_path)
    if not draws:
        print("[ERROR] No se pudo leer los datos")
        return

    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]
    
    print("="*85)
    print("       EVALUACIÓN COMPARATIVA: MODELO ACTUAL (POOL 12) VS MODELO MEJORADO (POOL 15 + PCR)")
    print(f" Sorteos Evaluados: {len(eval_draws)} (Últimos 50 sorteos de la modalidad Tradicional)")
    print("="*85)

    # Métricas Modelo Actual (Pool 12)
    current_pool_hits = []
    current_ticket_max_hits = []
    current_3plus = 0
    current_4plus = 0

    # Métricas Modelo Mejorado (Pool 15 + Co-ocurrencia Ponderada + Wheeling Cover 15)
    improved_pool_hits = []
    improved_ticket_max_hits = []
    improved_3plus = 0
    improved_4plus = 0

    start_time = time.time()

    for i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        target_nums = set(target_draw['numbers'])
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo or (d['sorteo'] == target_sorteo and d['modalidad'] != 'Tradicional')]
        if len(train_draws) < 100:
            continue

        # 1. FRECUENCIA RECIENTE Y LARGO PLAZO
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']:
                counts_60[n] += 1
                
        # 2. RECENCIA PONDERADA EXPONENCIAL
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']:
                recency_scores[n] += weight

        # 3. MATRIZ DE CO-OCURRENCIA EXPONENCIAL (PCR)
        cooccur_exp = np.zeros((46, 46))
        for t_idx, d in enumerate(train_draws[-40:]):
            w_t = np.exp((t_idx - 40) / 15.0)
            for u, v in itertools.combinations(d['numbers'], 2):
                cooccur_exp[u, v] += w_t
                cooccur_exp[v, u] += w_t

        cooccur_bonus = np.zeros(46)
        for n in range(46):
            cooccur_bonus[n] = np.sum(cooccur_exp[n])

        # 4. DELTA-DELAY MOMENTUM (DDM)
        last_seen = {n: -1 for n in range(46)}
        delays_list = defaultdict(list)
        for idx, d in enumerate(train_draws):
            for n in d['numbers']:
                if last_seen[n] != -1:
                    delays_list[n].append(idx - last_seen[n])
                last_seen[n] = idx
        
        ddm_scores = np.zeros(46)
        n_train = len(train_draws)
        for n in range(46):
            delay = n_train - 1 - last_seen[n] if last_seen[n] != -1 else n_train
            mean_d = np.mean(delays_list[n]) if delays_list[n] else 7.6
            std_d = np.std(delays_list[n]) if len(delays_list[n]) > 1 else 5.0
            # Normalizar z-score de delay
            z_delay = (delay - mean_d) / (std_d + 1e-5)
            # Premio moderado para números en su ventana esperada (-0.5 <= z <= 1.5)
            if -0.5 <= z_delay <= 1.5:
                ddm_scores[n] = 1.0 - abs(z_delay - 0.5)

        # ENSAMBLE PONDERADO BASELINE ACTUAL
        scores_current = np.zeros(46)
        for n in range(46):
            scores_current[n] = counts_60[n] * 0.4 + recency_scores[n] * 0.6

        ranking_current = np.argsort(scores_current)[::-1]
        pool_12 = set(ranking_current[:12])
        c_hits_pool = len(pool_12.intersection(target_nums))
        current_pool_hits.append(c_hits_pool)

        combos_12 = list(itertools.combinations(sorted(pool_12), 6))
        viables_12 = [c for c in combos_12 if es_ticket_valido(c)] or combos_12[:50]
        tickets_curr = select_wheel_coverage_tickets(viables_12, scores_current, n_tickets=3, target_subsize=4)
        c_max_t = max(len(set(t).intersection(target_nums)) for t in tickets_curr)
        current_ticket_max_hits.append(c_max_t)
        if c_max_t >= 3: current_3plus += 1
        if c_max_t >= 4: current_4plus += 1

        # ENSAMBLE PONDERADO MODELO MEJORADO
        scores_improved = np.zeros(46)
        for n in range(46):
            scores_improved[n] = (
                scores_current[n] * 0.5 + 
                (cooccur_bonus[n] / (np.max(cooccur_bonus) + 1e-5)) * 0.3 + 
                ddm_scores[n] * 0.2
            )

        ranking_improved = np.argsort(scores_improved)[::-1]
        pool_15 = set(ranking_improved[:15])
        imp_hits_pool = len(pool_15.intersection(target_nums))
        improved_pool_hits.append(imp_hits_pool)

        combos_15 = list(itertools.combinations(sorted(pool_15), 6))
        recent_nums = set()
        for d in train_draws[-5:]: recent_nums.update(d['numbers'])
        viables_15 = [c for c in combos_15 if es_ticket_valido(c, recent_numbers=recent_nums)] or combos_15[:100]
        
        # Selección Optimizada de Cobertura Wheeling en Pool 15
        tickets_imp = select_wheel_coverage_tickets(viables_15, scores_improved, n_tickets=3, target_subsize=4)
        imp_max_t = max(len(set(t).intersection(target_nums)) for t in tickets_imp)
        improved_ticket_max_hits.append(imp_max_t)
        if imp_max_t >= 3: improved_3plus += 1
        if imp_max_t >= 4: improved_4plus += 1

    elapsed = time.time() - start_time
    total = len(eval_draws)

    print("\n" + "="*85)
    print("                             RESULTADOS COMPARATIVOS")
    print("="*85)
    print(f" Métrica Evaluada                    | Modelo Actual (Pool 12) | Modelo Mejorado (Pool 15 + PCR + DDM)")
    print(" " + "-"*81)
    print(f" Promedio Aciertos en Pool           |        {np.mean(current_pool_hits):.2f} / 6        |         {np.mean(improved_pool_hits):.2f} / 6")
    print(f" Promedio Acierto Máx por Boleta     |        {np.mean(current_ticket_max_hits):.2f}            |         {np.mean(improved_ticket_max_hits):.2f}")
    print(f" % Sorteos con >= 3 Aciertos Boleta  |        {(current_3plus/total)*100:5.1f}%           |         {(improved_3plus/total)*100:5.1f}%")
    print(f" % Sorteos con >= 4 Aciertos Boleta  |        {(current_4plus/total)*100:5.1f}%           |         {(improved_4plus/total)*100:5.1f}%")
    print(" " + "-"*81)
    print(f" Tiempo total de evaluación: {elapsed:.2f} segundos")
    print("="*85)

if __name__ == '__main__':
    run_improvement_backtest()
