import csv
import numpy as np
import itertools
import os
import sys
import random
from collections import defaultdict

# Asegurar que el directorio del script esté en el path para imports locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from feature_engineering import freq_last_k, fft_energy, breakpoint_flag, delay_entropy, ewma_crossover

def leer_datos(archivo='c:/Users/Administrador/Desktop/lot/historico_quini_limpio.csv'):
    datos = []
    if not os.path.exists(archivo):
        print(f"Error: No existe el archivo {archivo}")
        return []
    with open(archivo, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bolillas = [int(row[f'B{i}']) for i in range(1, 7)]
                datos.append(bolillas)
            except ValueError:
                continue
    datos.reverse() # Cronológico (antiguos primero, recientes al final)
    return datos

def es_ticket_valido(ticket, recent_numbers=None):
    ticket = sorted(ticket)
    # 1. Consecutividad: máximo dos parejas consecutivas (relajado de 1 a 2)
    gaps_le_1 = sum(1 for i in range(1, len(ticket)) if ticket[i] - ticket[i-1] <= 1)
    if gaps_le_1 > 2:
        return False
    # 2. Spread: entre 15 y 45 (relajado de 18-44 a 15-45 para cubrir el 99% de los sorteos reales)
    spread = ticket[5] - ticket[0]
    if spread < 15 or spread > 45:
        return False
    # 3. Decenas: máximo 4 números por decena (relajado de 3 a 4)
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(count > 4 for count in decenas.values()):
        return False
    # 4. Paridad: entre 1 y 5 números pares
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares == 0 or pares == 6:
        return False
    # 5. Partición caliente/frío: exactamente 1 a 5 números calientes (relajado de 2-4 a 1-5)
    if recent_numbers is not None:
        recent_count = sum(1 for n in ticket if n in recent_numbers)
        if recent_count < 1 or recent_count > 5:
            return False
    return True

def extraer_features(datos_hist, max_num=45):
    n_sorteos = len(datos_hist)
    features = {}
    
    last_seen = {}
    delays_hist = defaultdict(list)
    
    for t_idx, sorteo in enumerate(datos_hist):
        for n in range(max_num + 1):
            if n in sorteo:
                if n in last_seen:
                    delay = t_idx - last_seen[n]
                    delays_hist[n].append(delay)
                last_seen[n] = t_idx
                
    mean_delays = {}
    std_delays = {}
    for n in range(max_num + 1):
        delays = delays_hist[n]
        mean_delays[n] = np.mean(delays) if delays else 7.6
        std_delays[n] = np.std(delays) if len(delays) > 1 else 5.0

    lags = {n: [0]*5 for n in range(max_num + 1)}
    for lag_k in range(1, 6):
        if n_sorteos >= lag_k:
            sorteo_k = datos_hist[-lag_k]
            for n in range(max_num + 1):
                if n in sorteo_k:
                    lags[n][lag_k-1] = 1

    ewma_3 = {n: 0.0 for n in range(max_num + 1)}
    ewma_10 = {n: 0.0 for n in range(max_num + 1)}
    ewma_30 = {n: 0.0 for n in range(max_num + 1)}
    
    alpha_3 = 2.0 / (3.0 + 1.0)
    alpha_10 = 2.0 / (10.0 + 1.0)
    alpha_30 = 2.0 / (30.0 + 1.0)
    
    for sorteo in datos_hist:
        for n in range(max_num + 1):
            val = 1.0 if n in sorteo else 0.0
            ewma_3[n] = alpha_3 * val + (1 - alpha_3) * ewma_3[n]
            ewma_10[n] = alpha_10 * val + (1 - alpha_10) * ewma_10[n]
            ewma_30[n] = alpha_30 * val + (1 - alpha_30) * ewma_30[n]

    cooc = defaultdict(lambda: defaultdict(int))
    markov_1_trans = defaultdict(lambda: defaultdict(int))
    markov_2_trans = defaultdict(lambda: defaultdict(int))
    markov_3_trans = defaultdict(lambda: defaultdict(int))
    
    for idx, sorteo in enumerate(datos_hist):
        for n1, n2 in itertools.combinations(sorteo, 2):
            cooc[n1][n2] += 1
            cooc[n2][n1] += 1
            
        if idx > 0:
            sorteo_prev = datos_hist[idx-1]
            for p in sorteo_prev:
                for c in sorteo:
                    markov_1_trans[p][c] += 1
                    
        if idx > 1:
            sorteo_prev2 = datos_hist[idx-2]
            for p in sorteo_prev2:
                for c in sorteo:
                    markov_2_trans[p][c] += 1
                    
        if idx > 2:
            sorteo_prev3 = datos_hist[idx-3]
            for p in sorteo_prev3:
                for c in sorteo:
                    markov_3_trans[p][c] += 1

    cooc_flat = {}
    for n1 in cooc:
        for n2 in cooc[n1]:
            cooc_flat[(n1, n2)] = cooc[n1][n2]
    max_cooc = max(cooc_flat.values()) if cooc_flat else 1
    
    ultimo_sorteo = datos_hist[-1] if n_sorteos > 0 else []
    penultimo_sorteo = datos_hist[-2] if n_sorteos > 1 else []
    ante_penultimo_sorteo = datos_hist[-3] if n_sorteos > 2 else []

    fft_vals_top3 = {}
    for n in range(max_num + 1):
        serie = np.zeros(n_sorteos)
        for i, sorteo in enumerate(datos_hist):
            if n in sorteo:
                serie[i] = 1.0
        media = np.mean(serie) if n_sorteos > 0 else 0
        serie_centrada = serie - media
        
        vals_cos = [0.0, 0.0, 0.0]
        if n_sorteos > 4:
            fft_vals = np.fft.fft(serie_centrada)
            fft_freqs = np.fft.fftfreq(n_sorteos)
            mitad = n_sorteos // 2
            amplitudes = np.abs(fft_vals[:mitad])
            freqs = fft_freqs[:mitad]
            
            if len(amplitudes) > 1:
                amplitudes[0] = 0
                top_indices = np.argsort(amplitudes)[-3:]
                for idx_p, idx_max in enumerate(reversed(top_indices)):
                    if idx_max < len(freqs) and amplitudes[idx_max] > 0:
                        freq_max = freqs[idx_max]
                        amp_max = amplitudes[idx_max]
                        fase = np.angle(fft_vals[idx_max])
                        t_next = n_sorteos
                        vals_cos[idx_p] = float(amp_max * np.cos(2 * np.pi * freq_max * t_next + fase))
        fft_vals_top3[n] = vals_cos

    for n in range(max_num + 1):
        delay = n_sorteos - 1 - last_seen.get(n, -1)
        if last_seen.get(n, -1) == -1:
            delay = n_sorteos
            
        delay_ratio = delay / mean_delays[n]
        
        cooc_last = sum(cooc[n].get(u, 0) for u in ultimo_sorteo) / (6.0 * max_cooc) if ultimo_sorteo else 0
        
        prob_markov1 = 0.0
        if ultimo_sorteo:
            for p in ultimo_sorteo:
                total_t = sum(markov_1_trans[p].values())
                if total_t > 0:
                    prob_markov1 += markov_1_trans[p][n] / total_t
            prob_markov1 /= 6.0
            
        prob_markov2 = 0.0
        if penultimo_sorteo:
            for p in penultimo_sorteo:
                total_t = sum(markov_2_trans[p].values())
                if total_t > 0:
                    prob_markov2 += markov_2_trans[p][n] / total_t
            prob_markov2 /= 6.0

        prob_markov3 = 0.0
        if ante_penultimo_sorteo:
            for p in ante_penultimo_sorteo:
                total_t = sum(markov_3_trans[p].values())
                if total_t > 0:
                    prob_markov3 += markov_3_trans[p][n] / total_t
            prob_markov3 /= 6.0

        # Compute additional engineered features for number n
        freq_last20 = freq_last_k(datos_hist, k=20, max_num=max_num)
        binary_series = np.array([1.0 if n in sorteo else 0.0 for sorteo in datos_hist])
        energy_n = fft_energy(binary_series)
        bp_flag = breakpoint_flag(datos_hist, window=30, max_num=max_num)
        
        # New advanced features
        delays_n = delays_hist.get(n, [])
        delay_ent = delay_entropy(delays_n, max_bins=5)
        crossover = ewma_crossover(binary_series, short_span=3, long_span=20)

        features[n] = [
            delay, delay_ratio, mean_delays[n], std_delays[n],
            ewma_3[n], ewma_10[n], ewma_30[n],
            lags[n][0], lags[n][1], lags[n][2], lags[n][3], lags[n][4],
            cooc_last, prob_markov1, prob_markov2, prob_markov3,
            fft_vals_top3[n][0], fft_vals_top3[n][1], fft_vals_top3[n][2],
            freq_last20[n],
            energy_n,
            bp_flag,
            delay_ent,
            crossover
        ]
        
    return features, cooc, max_cooc

features_cache = {}
def extraer_features_cached(datos, t, max_num=45):
    if t not in features_cache:
        features_cache[t] = extraer_features(datos[:t], max_num)
    return features_cache[t]

def preparar_dataset(datos, start_t, end_t):
    X = []
    y = []
    for t in range(start_t, end_t):
        real_t = set(datos[t])
        features_t, _, _ = extraer_features_cached(datos, t)
        for n in range(46):
            X.append(features_t[n])
            y.append(1 if n in real_t else 0)
    return np.array(X), np.array(y)

def prob_hits_poisson_binomial_flat(p):
    p0, p1, p2, p3, p4, p5 = p
    
    # Step 1 (p0)
    dp1_0 = 1.0 - p0
    dp1_1 = p0
    
    # Step 2 (p1)
    dp2_0 = dp1_0 * (1.0 - p1)
    dp2_1 = dp1_1 * (1.0 - p1) + dp1_0 * p1
    dp2_2 = dp1_1 * p1
    
    # Step 3 (p2)
    dp3_0 = dp2_0 * (1.0 - p2)
    dp3_1 = dp2_1 * (1.0 - p2) + dp2_0 * p2
    dp3_2 = dp2_2 * (1.0 - p2) + dp2_1 * p2
    dp3_3 = dp2_2 * p2
    
    # Step 4 (p3)
    dp4_0 = dp3_0 * (1.0 - p3)
    dp4_1 = dp3_1 * (1.0 - p3) + dp3_0 * p3
    dp4_2 = dp3_2 * (1.0 - p3) + dp3_1 * p3
    dp4_3 = dp3_3 * (1.0 - p3) + dp3_2 * p3
    dp4_4 = dp3_3 * p3
    
    # Step 5 (p4)
    dp5_0 = dp4_0 * (1.0 - p4)
    dp5_1 = dp4_1 * (1.0 - p4) + dp4_0 * p4
    dp5_2 = dp4_2 * (1.0 - p4) + dp4_1 * p4
    dp5_3 = dp4_3 * (1.0 - p4) + dp4_2 * p4
    dp5_4 = dp4_4 * (1.0 - p4) + dp4_3 * p4
    dp5_5 = dp4_4 * p4
    
    # Step 6 (p5)
    q5 = 1.0 - p5
    h3 = dp5_3 * q5 + dp5_2 * p5
    h4 = dp5_4 * q5 + dp5_3 * p5
    h5 = dp5_5 * q5 + dp5_4 * p5
    h6 = dp5_5 * p5
    
    return h3, h4, h5, h6

def predecir_tickets_para_sorteo(datos, t_pred, clf, pool_size=24, gamma=0.5, w3=0.1, w4=1.0, w5=10.0):
    features_t, cooc_i, max_cooc_i = extraer_features_cached(datos, t_pred)
    X_test = np.array([features_t[n] for n in range(46)])
    probs = clf.predict_proba(X_test)[:, 1]
    
    # 1. Compute empirical Markov transition boost based on history up to t_pred
    transitions = defaultdict(lambda: defaultdict(int))
    for t_idx in range(t_pred - 1):
        for n1 in datos[t_idx]:
            for n2 in datos[t_idx + 1]:
                transitions[n1][n2] += 1
                
    ultimo_sorteo = datos[t_pred - 1] if t_pred > 0 else []
    boost = np.zeros(46)
    if ultimo_sorteo:
        for p in ultimo_sorteo:
            total_transitions_from_p = sum(transitions[p].values())
            if total_transitions_from_p > 0:
                for n in range(46):
                    boost[n] += transitions[p][n] / total_transitions_from_p
        boost /= 6.0
        
    # Combine ML probability and Markov transition boost
    w_ml = 0.7
    w_trans = 0.3
    combined_probs = (w_ml * probs) + (w_trans * boost)
    
    # 2. Extract recent numbers from the last 5 draws for the partition filter
    recent_numbers = set()
    for d in datos[max(0, t_pred - 5):t_pred]:
        recent_numbers.update(d)
        
    ranking = [(n, combined_probs[n]) for n in range(46)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    top_k = sorted([x[0] for x in ranking[:pool_size]])
    
    todas_combinaciones = list(itertools.combinations(top_k, 6))
    
    # Filter with hot/cold partition constraint
    boletos_filtrados = [combo for combo in todas_combinaciones if es_ticket_valido(combo, recent_numbers)]
    if not boletos_filtrados:
        # Fallback to basic validity filters (relax partition filter)
        boletos_filtrados = [combo for combo in todas_combinaciones if es_ticket_valido(combo)]
    if not boletos_filtrados:
        # Absolute fallback
        boletos_filtrados = todas_combinaciones[:100]
        
    # 3. Sequential selection of 3 tickets using Poisson Binomial and probability discounting
    tickets_optimos = []
    current_probs = combined_probs.copy()
    
    for _ in range(3):
        best_ticket = None
        best_fitness = -1.0
        
        for boleto in boletos_filtrados:
            c_probs = [current_probs[n] for n in boleto]
            h3, h4, h5, h6 = prob_hits_poisson_binomial_flat(c_probs)
            
            # Simple unrolled equivalent of cumulative probabilities:
            # p3_plus = h3 + h4 + h5 + h6
            # p4_plus = h4 + h5 + h6
            # p5_plus = h5 + h6
            # fitness = w3 * p3_plus + w4 * p4_plus + w5 * p5_plus
            # Using pre-computed algebraic simplified weights:
            # fitness = w3 * h3 + (w3 + w4) * h4 + (w3 + w4 + w5) * (h5 + h6)
            fitness = w3 * h3 + (w3 + w4) * h4 + (w3 + w4 + w5) * (h5 + h6)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_ticket = boleto
                
        if best_ticket is not None:
            tickets_optimos.append(best_ticket)
            # Discount the probabilities of the selected numbers for diversity
            for n in best_ticket:
                current_probs[n] *= gamma
        else:
            # Fallback if no ticket could be chosen
            tickets_optimos.append(boletos_filtrados[0])
            
    return tickets_optimos, combined_probs

def ejecutar_backtest_acumulativo(datos, start_idx=35, pool_size=24, gamma=0.5, w3=0.1, w4=1.0, w5=10.0):
    n_sorteos = len(datos)
    print("==================================================")
    print(f"INICIANDO SIMULACION COMPLETA PASO A PASO (WALK-FORWARD)")
    print(f"Desde Sorteo Nro {start_idx} hasta {n_sorteos-1}")
    print(f"Total a evaluar: {n_sorteos - start_idx} sorteos historicos")
    print("==================================================\n")

    hits_distribution = {i: 0 for i in range(7)}
    totales_tickets_evaluados = 0
    control_hits_distribution = {i: 0 for i in range(7)}
    
    # ---------- Ensemble Setup ----------
    from grid_search_util import grid_search
    # Define modest hyper‑parameter grids for each algorithm
    xgb_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'n_estimators': [100, 200],
    }
    lgbm_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'n_estimators': [100, 200],
    }
    cat_grid = {
        'depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'iterations': [200, 400],
    }
    best_xgb_params = None
    best_lgbm_params = None
    best_cat_params = None
    
    # Pre-populate training lists up to start_idx
    X_train_list = []
    y_train_list = []
    for t in range(0, start_idx):
        real_t = set(datos[t])
        features_t, _, _ = extraer_features_cached(datos, t)
        for n in range(46):
            X_train_list.append(features_t[n])
            y_train_list.append(1 if n in real_t else 0)
            
    for i in range(start_idx, n_sorteos):
        real_draw = set(datos[i])
        
        # Convert training lists to numpy arrays for fitting
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        # Tune hyperparameters every 50 steps or on the first step
        if best_xgb_params is None or (i - start_idx) % 50 == 0:
            best_xgb_params, _ = grid_search(datos, xgb_grid, XGBClassifier, preparar_dataset, i, 
                                             fixed_kwargs={'use_label_encoder': False, 'eval_metric': 'logloss', 'verbosity': 0, 'random_state': 42}, 
                                             max_iter=5)
            best_lgbm_params, _ = grid_search(datos, lgbm_grid, LGBMClassifier, preparar_dataset, i, 
                                              fixed_kwargs={'verbose': -1, 'random_state': 42}, 
                                              max_iter=5)
            best_cat_params, _ = grid_search(datos, cat_grid, CatBoostClassifier, preparar_dataset, i, 
                                             fixed_kwargs={'logging_level': 'Silent', 'random_state': 42}, 
                                             max_iter=5)
            
        # Train models on the full training window (silent settings)
        model_xgb = XGBClassifier(**best_xgb_params, use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42)
        model_lgbm = LGBMClassifier(**best_lgbm_params, verbose=-1, random_state=42)
        model_cat = CatBoostClassifier(**best_cat_params, logging_level='Silent', random_state=42)
        
        model_xgb.fit(X_train, y_train)
        model_lgbm.fit(X_train, y_train)
        model_cat.fit(X_train, y_train)
        
        # Simple average‑probability ensemble wrapper
        class AvgEnsemble:
            def __init__(self, models):
                self.models = models
            def predict_proba(self, X):
                probs = [m.predict_proba(X)[:, 1] for m in self.models]
                avg = np.mean(probs, axis=0)
                # Return a 2‑column array to mimic sklearn interface
                return np.vstack([1 - avg, avg]).T
        clf = AvgEnsemble([model_xgb, model_lgbm, model_cat])
        # ------------------------------------------------
        
        tickets, _ = predecir_tickets_para_sorteo(
            datos, i, clf,
            pool_size=pool_size,
            gamma=gamma,
            w3=w3,
            w4=w4,
            w5=w5
        )
        
        # Evaluar aciertos del Modelo ML
        for idx, b in enumerate(tickets):
            hits = len(set(b).intersection(real_draw))
            hits_distribution[hits] += 1
            totales_tickets_evaluados += 1
            
        # Evaluar aciertos del Control Aleatorio
        for _ in range(3):
            while True:
                rnd_ticket = random.sample(range(46), 6)
                if es_ticket_valido(rnd_ticket):
                    break
            hits_rnd = len(set(rnd_ticket).intersection(real_draw))
            control_hits_distribution[hits_rnd] += 1
            
        # Append draw i features and labels to the training set for the next iteration (i+1)
        features_i, _, _ = extraer_features_cached(datos, i)
        for n in range(46):
            X_train_list.append(features_i[n])
            y_train_list.append(1 if n in real_draw else 0)
            
        if (i - start_idx + 1) % 50 == 0 or i == n_sorteos - 1:
            progreso = ((i - start_idx + 1) / (n_sorteos - start_idx)) * 100
            print(f"Progreso: {progreso:.1f}% ({i - start_idx + 1}/{n_sorteos - start_idx} sorteos simulados)")

    ml_3_plus = sum(hits_distribution[h] for h in range(3, 7))
    ctrl_3_plus = sum(control_hits_distribution[h] for h in range(3, 7))
    
    print("\n==================================================")
    print("RESULTADOS FINALES DE LA SIMULACION COMPLETA")
    print(f"Evaluados: {n_sorteos - start_idx} sorteos | Total tickets: {totales_tickets_evaluados}")
    print("==================================================")
    print("Aciertos   | Modelo ML (Freq) | Control Aleatorio (Freq)")
    print("-----------+------------------+---------------------------")
    for h in range(7):
        pct_ml = (hits_distribution[h] / totales_tickets_evaluados) * 100
        pct_ctrl = (control_hits_distribution[h] / totales_tickets_evaluados) * 100
        print(f" {h} aciertos | {hits_distribution[h]:4d} ({pct_ml:6.2f}%)   | {control_hits_distribution[h]:4d} ({pct_ctrl:6.2f}%)")
    print("-----------+------------------+---------------------------")
    print(f" 3+ Hits   | {ml_3_plus:4d} ({(ml_3_plus/totales_tickets_evaluados)*100:6.2f}%)   | {ctrl_3_plus:4d} ({(ctrl_3_plus/totales_tickets_evaluados)*100:6.2f}%)")
    print("==================================================")
    
    # Calcular promedio de aciertos por ticket
    total_hits_ml = sum(h * count for h, count in hits_distribution.items())
    total_hits_ctrl = sum(h * count for h, count in control_hits_distribution.items())
    prom_ml = total_hits_ml / totales_tickets_evaluados
    prom_ctrl = total_hits_ctrl / totales_tickets_evaluados
    print(f"Promedio de aciertos por ticket (Modelo ML): {prom_ml:.4f}")
    print(f"Promedio de aciertos por ticket (Aleatorio): {prom_ctrl:.4f}")
    if prom_ctrl > 0:
        print(f"Ventaja relativa frente al azar: {(prom_ml / prom_ctrl):.2f}x")
    else:
        print("Ventaja relativa frente al azar: N/A (Control Aleatorio obtuvo 0 aciertos)")
    print("==================================================")

if __name__ == '__main__':
    datos = leer_datos()
    if not datos:
        sys.exit(1)
    # Ejecutamos la simulación acumulativa desde el sorteo 35 hasta el último (sorteo 646)
    ejecutar_backtest_acumulativo(datos, start_idx=35)
