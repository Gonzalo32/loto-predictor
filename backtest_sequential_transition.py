import csv
import os
import sys
import random
import numpy as np
import itertools
import warnings
from collections import defaultdict

# Asegurar que el directorio del script esté en el path para imports locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# 1. Load the complete sequential data
def load_data(filepath="c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv"):
    draws = []
    # Columns: Sorteo,Fecha,Modalidad,B1,B2,B3,B4,B5,B6
    if not os.path.exists(filepath):
        # Fallback to local workspace folder
        filepath = "historico_quini_completo.csv"
        
    if not os.path.exists(filepath):
        print(f"Error: No se encontró el archivo de datos históricos: {filepath}")
        return []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sorteo = int(row['Sorteo'])
                fecha = row['Fecha']
                modalidad = row['Modalidad'].strip()
                # Parse balls B1 to B6
                numbers = sorted([int(row[f'B{i}']) for i in range(1, 7)])
                draws.append({
                    'sorteo': sorteo,
                    'fecha': fecha,
                    'modalidad': modalidad,
                    'numbers': numbers
                })
            except Exception:
                continue
                
    # Sort chronologically: Sorteo ascending, then modality order
    modality_order = {
        'Tradicional': 1,
        'Segunda': 2,
        'Revancha': 3,
        'SiempreSale': 4
    }
    draws.sort(key=lambda x: (x['sorteo'], modality_order.get(x['modalidad'], 9)))
    return draws

# 2. Check ticket validity
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

# 3. Probability evaluations for Poisson Binomial
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

# 4. Sequential Feature Extraction
features_cache = {}

def extract_features_for_step(draws, t, transition_counts, transition2_counts, transition_mod_counts):
    target_mod = draws[t]['modalidad']
    
    # Target modality numeric encoding
    mod_order = {'Tradicional': 0, 'Segunda': 1, 'Revancha': 2, 'SiempreSale': 3}
    mod_numeric = mod_order.get(target_mod, 0)
    
    history = draws[:t]
    n_history = len(history)
    
    # Precompute last seen indices and delays
    last_seen = {n: -1 for n in range(46)}
    delays_hist = defaultdict(list)
    for idx, d in enumerate(history):
        for n in d['numbers']:
            if last_seen[n] != -1:
                delays_hist[n].append(idx - last_seen[n])
            last_seen[n] = idx
            
    mean_delays = {}
    std_delays = {}
    for n in range(46):
        delays = delays_hist[n]
        mean_delays[n] = np.mean(delays) if delays else 7.6
        std_delays[n] = np.std(delays) if len(delays) > 1 else 5.0

    # Lags in sequence
    lags = {n: [0]*8 for n in range(46)}
    for lag_k in range(1, 9):
        if n_history >= lag_k:
            d_k = history[-lag_k]
            for n in d_k['numbers']:
                lags[n][lag_k-1] = 1

    # EWMA calculations
    ewma_3 = {n: 0.0 for n in range(46)}
    ewma_10 = {n: 0.0 for n in range(46)}
    ewma_30 = {n: 0.0 for n in range(46)}
    alpha_3 = 2.0 / (3.0 + 1.0)
    alpha_10 = 2.0 / (10.0 + 1.0)
    alpha_30 = 2.0 / (30.0 + 1.0)
    
    for d in history:
        for n in range(46):
            val = 1.0 if n in d['numbers'] else 0.0
            ewma_3[n] = alpha_3 * val + (1 - alpha_3) * ewma_3[n]
            ewma_10[n] = alpha_10 * val + (1 - alpha_10) * ewma_10[n]
            ewma_30[n] = alpha_30 * val + (1 - alpha_30) * ewma_30[n]

    # Modality-specific frequency and delay
    mod_history = [d for d in history if d['modalidad'] == target_mod]
    n_mod_history = len(mod_history)
    
    last_seen_mod = {n: -1 for n in range(46)}
    for idx, d in enumerate(mod_history):
        for n in d['numbers']:
            last_seen_mod[n] = idx
            
    features = {}
    
    # Previous draw info
    d_prev = history[-1] if n_history > 0 else None
    d_prev2 = history[-2] if n_history > 1 else None
    
    for n in range(46):
        # 1. Delays
        delay = t - 1 - last_seen[n] if last_seen[n] != -1 else t
        delay_ratio = delay / (mean_delays[n] + 1e-5)
        
        # 2. Modality specific delay
        mod_delay = n_mod_history - 1 - last_seen_mod[n] if last_seen_mod[n] != -1 else n_mod_history
        
        # 3. Transition scores (Laplace smoothed)
        trans_score = 0.0
        if d_prev:
            key = (d_prev['modalidad'], target_mod)
            for x in d_prev['numbers']:
                num = transition_counts[key][x].get(n, 0) + 0.1
                denom = sum(transition_counts[key][x].values()) + 46 * 0.1
                trans_score += num / denom
            trans_score /= 6.0
            
        trans_score_lag2 = 0.0
        if d_prev2:
            key = (d_prev2['modalidad'], target_mod)
            for x in d_prev2['numbers']:
                num = transition2_counts[key][x].get(n, 0) + 0.1
                denom = sum(transition2_counts[key][x].values()) + 46 * 0.1
                trans_score_lag2 += num / denom
            trans_score_lag2 /= 6.0
            
        # 3.b Modality-specific transition score (Laplace smoothed)
        trans_mod_score = 0.0
        if len(mod_history) > 0:
            d_prev_mod = mod_history[-1]
            for x in d_prev_mod['numbers']:
                num = transition_mod_counts[target_mod][x].get(n, 0) + 0.1
                denom = sum(transition_mod_counts[target_mod][x].values()) + 46 * 0.1
                trans_mod_score += num / denom
            trans_mod_score /= 6.0
            
        # 4. Frequencies
        freq_20 = sum(1 for d in history[-20:] if n in d['numbers']) / 20.0 if n_history >= 20 else sum(1 for d in history if n in d['numbers']) / (n_history + 1e-5)
        freq_50 = sum(1 for d in history[-50:] if n in d['numbers']) / 50.0 if n_history >= 50 else sum(1 for d in history if n in d['numbers']) / (n_history + 1e-5)
        freq_100 = sum(1 for d in history[-100:] if n in d['numbers']) / 100.0 if n_history >= 100 else sum(1 for d in history if n in d['numbers']) / (n_history + 1e-5)
        
        mod_freq_10 = sum(1 for d in mod_history[-10:] if n in d['numbers']) / 10.0 if n_mod_history >= 10 else sum(1 for d in mod_history if n in d['numbers']) / (n_mod_history + 1e-5)
        mod_freq_30 = sum(1 for d in mod_history[-30:] if n in d['numbers']) / 30.0 if n_mod_history >= 30 else sum(1 for d in mod_history if n in d['numbers']) / (n_mod_history + 1e-5)
        
        # 5. Advanced statistics
        binary_series = np.array([1.0 if n in d['numbers'] else 0.0 for d in history])
        crossover = 1.0 if ewma_3[n] > ewma_10[n] else 0.0
        
        delays_n = delays_hist[n]
        if len(delays_n) < 3:
            entropy = 0.0
        else:
            counts, _ = np.histogram(delays_n, bins=5)
            probs = counts / (np.sum(counts) + 1e-5)
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs)))
            
        if n_history == 0:
            fft_energy = 0.0
        else:
            mean_series = np.mean(binary_series)
            fft_vals = np.fft.fft(binary_series - mean_series)
            fft_energy = float(np.sum(np.abs(fft_vals) ** 2))
            
        features[n] = [
            delay, delay_ratio, mean_delays[n], std_delays[n],
            mod_delay,
            trans_score, trans_score_lag2, trans_mod_score,
            lags[n][0], lags[n][1], lags[n][2], lags[n][3], lags[n][4], lags[n][5], lags[n][6], lags[n][7],
            ewma_3[n], ewma_10[n], ewma_30[n],
            freq_20, freq_50, freq_100,
            mod_freq_10, mod_freq_30,
            mod_numeric,
            crossover, entropy, fft_energy
        ]
        
    return features

features_cache = {}

def get_features_for_step_cached(draws, t, transition_counts, transition2_counts, transition_mod_counts):
    if t not in features_cache:
        features_cache[t] = extract_features_for_step(draws, t, transition_counts, transition2_counts, transition_mod_counts)
    return features_cache[t]

# 5. Dynamic update of transitions count dictionaries
def update_transition_counts(draws, i, transition_counts, transition2_counts, transition_mod_counts):
    if i > 0:
        d_prev = draws[i-1]
        d_curr = draws[i]
        key = (d_prev['modalidad'], d_curr['modalidad'])
        for x in d_prev['numbers']:
            for y in d_curr['numbers']:
                transition_counts[key][x][y] += 1
                
    if i > 1:
        d_prev2 = draws[i-2]
        d_curr = draws[i]
        key = (d_prev2['modalidad'], d_curr['modalidad'])
        for x in d_prev2['numbers']:
            for y in d_curr['numbers']:
                transition2_counts[key][x][y] += 1
                
    # Modality-specific chronological transitions (Tradicional Sorteo T-1 -> Tradicional Sorteo T, etc.)
    d_curr = draws[i]
    target_mod = d_curr['modalidad']
    for prev_idx in range(i-1, -1, -1):
        if draws[prev_idx]['modalidad'] == target_mod:
            d_prev_mod = draws[prev_idx]
            for x in d_prev_mod['numbers']:
                for y in d_curr['numbers']:
                    transition_mod_counts[target_mod][x][y] += 1
            break

# 6. Sequential selection of 3 tickets using Poisson Binomial and probability discounting
def predecir_tickets_para_sorteo(probs, pool_size=24, gamma=0.5, w3=0.1, w4=1.0, w5=10.0, recent_numbers=None, custom_ranking=None):
    if custom_ranking is not None:
        top_k = sorted(custom_ranking[:pool_size])
    else:
        ranking = [(n, probs[n]) for n in range(46)]
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
        
    tickets_optimos = []
    current_probs = probs.copy()
    
    for _ in range(3):
        best_ticket = None
        best_fitness = -1.0
        
        for boleto in boletos_filtrados:
            c_probs = [current_probs[n] for n in boleto]
            h3, h4, h5, h6 = prob_hits_poisson_binomial_flat(c_probs)
            
            # fitness score calculation
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
            
    return tickets_optimos

# 7. Main Walk-Forward Backtesting Loop
def ejecutar_backtest_secuencial(draws, start_idx=200, train_interval=20):
    n_draws = len(draws)
    print("==================================================")
    print("INICIANDO SIMULACION WALK-FORWARD SECUENCIAL")
    print(f"Desde paso secuencial {start_idx} hasta {n_draws-1}")
    print(f"Total a evaluar: {n_draws - start_idx} sub-sorteos históricos")
    print("==================================================\n")
    
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    transition2_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    transition_mod_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Initialize transition matrices up to start_idx - 1
    print("Inicializando matrices de transición...")
    for i in range(start_idx):
        update_transition_counts(draws, i, transition_counts, transition2_counts, transition_mod_counts)
    print("Matrices inicializadas.")
    
    # Metric accumulators
    hits_distribution = {i: 0 for i in range(7)}
    totales_tickets_evaluados = 0
    control_hits_distribution = {i: 0 for i in range(7)}
    
    # Ranking metrics
    ranking_hits = {6: 0, 12: 0, 18: 0}
    
    # Modality specific hits
    modality_hits = defaultdict(list)
    
    # Setup ML models
    clf_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1)
    clf_lgb = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15, verbose=-1, random_state=42, n_jobs=-1)
    clf_cat = CatBoostClassifier(iterations=150, learning_rate=0.05, depth=4, logging_level='Silent', random_state=42, thread_count=-1)
    
    trained_at = -1
    
    # Pre-cache training features up to start_idx to speed up walk-forward
    print("Pre-calculando características iniciales...")
    for i in range(20, start_idx):
        get_features_for_step_cached(draws, i, transition_counts, transition2_counts, transition_mod_counts)
    print("Características iniciales pre-calculadas.")
    
    for t in range(start_idx, n_draws):
        target_mod = draws[t]['modalidad']
        real_draw = set(draws[t]['numbers'])
        
        # 1. Train models periodically (every train_interval steps)
        if trained_at == -1 or (t - trained_at) >= train_interval:
            # Build training dataset from step 20 up to t-1
            X_train_list = []
            y_train_list = []
            for i in range(20, t):
                feats_i = get_features_for_step_cached(draws, i, transition_counts, transition2_counts, transition_mod_counts)
                real_draw_i = set(draws[i]['numbers'])
                for n in range(46):
                    X_train_list.append(feats_i[n])
                    y_train_list.append(1 if n in real_draw_i else 0)
                    
            X_train = np.array(X_train_list)
            y_train = np.array(y_train_list)
            
            # Fit models
            clf_xgb.fit(X_train, y_train)
            clf_lgb.fit(X_train, y_train)
            clf_cat.fit(X_train, y_train)
            
            trained_at = t
            
        # 2. Extract features for step t
        feats_t = get_features_for_step_cached(draws, t, transition_counts, transition2_counts, transition_mod_counts)
        X_test = np.array([feats_t[n] for n in range(46)])
        
        # 3. Predict probabilities
        p_xgb = clf_xgb.predict_proba(X_test)[:, 1]
        p_lgb = clf_lgb.predict_proba(X_test)[:, 1]
        p_cat = clf_cat.predict_proba(X_test)[:, 1]
        probs_t = (p_xgb + p_lgb + p_cat) / 3.0
        
        # Borda Count Rank Fusion for choosing the pool
        borda = defaultdict(float)
        for p_model in [p_xgb, p_lgb, p_cat]:
            ranking_model = sorted(range(46), key=lambda x: p_model[x], reverse=True)
            for rank, n in enumerate(ranking_model):
                borda[n] += (45 - rank)
        borda_ranking = [n for n, score in sorted(borda.items(), key=lambda x: x[1], reverse=True)]
        
        # 4. Rank probabilities and evaluate top list coverage using Borda Count Ranking
        ranking = borda_ranking
        
        r6_hits = len(real_draw.intersection(ranking[:6]))
        r12_hits = len(real_draw.intersection(ranking[:12]))
        r18_hits = len(real_draw.intersection(ranking[:18]))
        
        ranking_hits[6] += r6_hits
        ranking_hits[12] += r12_hits
        ranking_hits[18] += r18_hits
        
        # 5. Extract recent numbers from the last 5 sequence steps for hot/cold filter
        recent_numbers = set()
        for d in draws[max(0, t - 5):t]:
            recent_numbers.update(d['numbers'])
            
        # 6. Generate optimized tickets using average probabilities but with custom Borda ranking pool
        tickets = predecir_tickets_para_sorteo(probs_t, pool_size=24, gamma=0.5, w3=0.1, w4=1.0, w5=10.0, recent_numbers=recent_numbers, custom_ranking=borda_ranking)
        
        # 7. Evaluate tickets against actual draw numbers
        for ticket in tickets:
            hits = len(set(ticket).intersection(real_draw))
            hits_distribution[hits] += 1
            totales_tickets_evaluados += 1
            modality_hits[target_mod].append(hits)
            
        # 8. Random Control Evaluation (3 tickets)
        for _ in range(3):
            rnd_ticket = random.sample(range(46), 6)
            for _intentos in range(1000):
                if es_ticket_valido(rnd_ticket):
                    break
                rnd_ticket = random.sample(range(46), 6)
            hits_rnd = len(set(rnd_ticket).intersection(real_draw))
            control_hits_distribution[hits_rnd] += 1
            
        # 9. Update transitions with draw t numbers
        update_transition_counts(draws, t, transition_counts, transition2_counts, transition_mod_counts)
        
        # Progress log
        if (t - start_idx + 1) % 100 == 0 or t == n_draws - 1:
            prog = ((t - start_idx + 1) / (n_draws - start_idx)) * 100
            print(f"Progreso: {prog:.1f}% ({t - start_idx + 1}/{n_draws - start_idx} sorteos simulados)")
            
    # Calculate global metrics
    ml_3_plus = sum(hits_distribution[h] for h in range(3, 7))
    ctrl_3_plus = sum(control_hits_distribution[h] for h in range(3, 7))
    
    total_hits_ml = sum(h * count for h, count in hits_distribution.items())
    total_hits_ctrl = sum(h * count for h, count in control_hits_distribution.items())
    prom_ml = total_hits_ml / totales_tickets_evaluados
    prom_ctrl = total_hits_ctrl / totales_tickets_evaluados
    
    print("\n==================================================")
    print("RESULTADOS FINALES DE LA SIMULACION SECUENCIAL")
    print(f"Evaluados: {n_draws - start_idx} sub-sorteos | Total tickets: {totales_tickets_evaluados}")
    print("==================================================")
    print("Aciertos   | Modelo Secuencial (Freq) | Control Aleatorio (Freq)")
    print("-----------+--------------------------+---------------------------")
    for h in range(7):
        pct_ml = (hits_distribution[h] / totales_tickets_evaluados) * 100
        pct_ctrl = (control_hits_distribution[h] / totales_tickets_evaluados) * 100
        print(f" {h} aciertos | {hits_distribution[h]:6d} ({pct_ml:6.2f}%)       | {control_hits_distribution[h]:6d} ({pct_ctrl:6.2f}%)")
    print("-----------+--------------------------+---------------------------")
    print(f" 3+ Hits   | {ml_3_plus:6d} ({(ml_3_plus/totales_tickets_evaluados)*100:6.2f}%)       | {ctrl_3_plus:6d} ({(ctrl_3_plus/totales_tickets_evaluados)*100:6.2f}%)")
    print("==================================================")
    
    print(f"Promedio de aciertos por ticket (Modelo ML): {prom_ml:.4f}")
    print(f"Promedio de aciertos por ticket (Aleatorio): {prom_ctrl:.4f}")
    if prom_ctrl > 0:
        print(f"Ventaja relativa frente al azar: {(prom_ml / prom_ctrl):.2f}x")
    print("==================================================")
    
    print("\n--- Precisión de Cobertura en Rankings de Probabilidad ---")
    n_eval = n_draws - start_idx
    print(f"Promedio de aciertos en Top 6:   {ranking_hits[6] / n_eval:.2f} / 6.00")
    print(f"Promedio de aciertos en Top 12:  {ranking_hits[12] / n_eval:.2f} / 6.00")
    print(f"Promedio de aciertos en Top 18:  {ranking_hits[18] / n_eval:.2f} / 6.00")
    
    print("\n--- Rendimiento Promedio de Aciertos por Modalidad ---")
    for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
        hits_mod = modality_hits[mod]
        if hits_mod:
            prom_mod = np.mean(hits_mod)
            print(f"Modalidad {mod:12s} | Promedio aciertos: {prom_mod:.4f}")
    print("==================================================")

if __name__ == '__main__':
    datos = load_data()
    if not datos:
        sys.exit(1)
    
    # Ejecutamos la simulación walk-forward secuencial a partir del paso 200
    ejecutar_backtest_secuencial(datos, start_idx=2000, train_interval=20)
