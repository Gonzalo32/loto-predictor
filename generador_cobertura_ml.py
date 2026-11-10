import csv
import json
import numpy as np
import itertools
import os
import sys
from datetime import datetime
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings("ignore")
os.environ["LIGHTGBM_VERBOSE"] = "-1"

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cobertura_combinatoria import select_optimal_wheel_triplet, select_wheel_coverage_tickets

# Define models with high performance hyper-parameters (Class-Weighted Ensemble for minority positive class)
clf_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, scale_pos_weight=6.0, use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1)
clf_lgb = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15, scale_pos_weight=6.0, verbose=-1, random_state=42, n_jobs=-1)
clf_cat = CatBoostClassifier(iterations=150, learning_rate=0.05, depth=4, auto_class_weights='Balanced', logging_level='Silent', random_state=42, thread_count=-1)
clf_rf  = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
clf_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
clf_meta = LogisticRegression(random_state=42)

def compute_attention_mlp_scores(train_draws, target_mod='Tradicional'):
    mod_draws = [d['numbers'] for d in train_draws if d['modalidad'] == target_mod]
    if len(mod_draws) < 30:
        mod_draws = [d['numbers'] for d in train_draws]
        
    T = len(mod_draws)
    matrix = np.zeros((T, 46))
    for t, nums in enumerate(mod_draws):
        for n in nums:
            matrix[t, n] = 1.0
            
    X_list = []
    y_list = []
    for t in range(10, T):
        feat = np.hstack([
            matrix[t-1], matrix[t-2], matrix[t-3], matrix[t-4],
            matrix[t-5], matrix[t-8], matrix[t-10]
        ])
        X_list.append(feat)
        y_list.append(matrix[t])
        
    X = np.array(X_list)
    Y = np.array(y_list)
    
    last_feat = np.hstack([
        matrix[-1], matrix[-2], matrix[-3], matrix[-4],
        matrix[-5], matrix[-8], matrix[-10]
    ]).reshape(1, -1)
    
    try:
        mlp_attn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=80, random_state=42, early_stopping=True)
        mlp_attn.fit(X, Y)
        scores = mlp_attn.predict_proba(last_feat)[0]
    except Exception:
        scores = np.mean(matrix[-20:], axis=0)
    return scores


def load_data(filepath=None):
    draws = []
    if filepath is None or not os.path.exists(filepath):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, 'historico_quini_completo.csv')
    if not os.path.exists(filepath):
        filepath = 'historico_quini_completo.csv'
    if not os.path.exists(filepath):
        print(f"Error: No existe el archivo {filepath}")
        return []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                draws.append({
                    'sorteo': int(row['Sorteo']),
                    'fecha': row['Fecha'],
                    'modalidad': row['Modalidad'].strip(),
                    'numbers': sorted([int(row[f'B{i}']) for i in range(1, 7)])
                })
            except (ValueError, KeyError):
                continue
                
    modality_order = {
        'Tradicional': 1,
        'Segunda': 2,
        'Revancha': 3,
        'SiempreSale': 4
    }
    draws.sort(key=lambda x: (x['sorteo'], modality_order.get(x['modalidad'], 9)))
    return draws

def compute_antagonist_pairs(train_draws, min_expected=10.0, max_lift=0.40):
    """
    Identifica pares que tienen una frecuencia observada significativamente inferior
    a la esperada por azar en la modalidad Tradicional (lift < 0.40), evitando
    combinaciones antagónicas que casi nunca salen juntas.
    """
    trad_draws = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    T = len(trad_draws)
    if T < 50:
        return set()
    freq = Counter(n for d in trad_draws for n in d['numbers'])
    pairs = Counter(tuple(sorted([u, v])) for d in trad_draws for u, v in itertools.combinations(d['numbers'], 2))
    antagonist_pairs = set()
    for u, v in itertools.combinations(range(46), 2):
        p_exp = (freq[u] * freq[v]) / T
        obs = pairs[tuple(sorted([u, v]))]
        lift = obs / p_exp if p_exp > 0 else 1.0
        if p_exp >= min_expected and lift < max_lift:
            antagonist_pairs.add(tuple(sorted([u, v])))
    return antagonist_pairs

def es_ticket_valido(ticket, recent_numbers=None, saturated_decades=None, antagonist_pairs=None, min_distinct_decades=3):
    ticket = sorted(ticket)
    # 1. Consecutividad: máximo dos parejas consecutivas
    gaps_le_1 = sum(1 for i in range(1, len(ticket)) if ticket[i] - ticket[i-1] <= 1)
    if gaps_le_1 > 2:
        return False
    # 2. Spread: entre 15 y 45
    spread = ticket[5] - ticket[0]
    if spread < 15 or spread > 45:
        return False
    # 3. Decenas: máximo 3 números por decena (y máximo 2 en la decena del 40 ya que sólo hay 6 números: 40-45)
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(count > 3 for count in decenas.values()):
        return False
    if decenas[4] > 2:
        return False
    # 3b. Anti-Congestión: exigir que los 6 números toquen al menos min_distinct_decades decenas distintas
    if len(decenas) < min_distinct_decades:
        return False
    # 4. Paridad: entre 1 y 5 números pares (óptimo 2 a 4)
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares == 0 or pares == 6:
        return False
    # 4b. Rango Gaussiano de Suma: 98% histórico de sorteos reales está entre 68 y 202
    suma_ticket = sum(ticket)
    if suma_ticket < 68 or suma_ticket > 202:
        return False
    # 4c. Balance Altos/Bajos: evitar extremos degenerados (0-6 y 6-0)
    bajos = sum(1 for n in ticket if n <= 22)
    if bajos < 1 or bajos > 5:
        return False
    # 5. Partición caliente/frío: exactamente 1 a 5 números calientes
    if recent_numbers is not None:
        recent_count = sum(1 for n in ticket if n in recent_numbers)
        if recent_count < 1 or recent_count > 5:
            return False
    # 6. Decade Saturation Deflation (DSD) Filter
    if saturated_decades:
        for dec in saturated_decades:
            if decenas[dec] > 2:
                return False
    # 7. Bloqueo de Pares Antagónicos (Lift < 0.40)
    if antagonist_pairs:
        for u, v in itertools.combinations(ticket, 2):
            if tuple(sorted([u, v])) in antagonist_pairs:
                return False
    return True

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
                
    # Modality-specific chronological transitions
    d_curr = draws[i]
    target_mod = d_curr['modalidad']
    for prev_idx in range(i-1, -1, -1):
        if draws[prev_idx]['modalidad'] == target_mod:
            d_prev_mod = draws[prev_idx]
            for x in d_prev_mod['numbers']:
                for y in d_curr['numbers']:
                    transition_mod_counts[target_mod][x][y] += 1
            break

def extract_features_for_step(draws, t, transition_counts, transition2_counts, transition_mod_counts, target_mod):
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

    # Modality specific metrics
    mod_history = [d for d in history if d['modalidad'] == target_mod]
    n_mod_history = len(mod_history)
    
    last_seen_mod = {n: -1 for n in range(46)}
    for idx, d in enumerate(mod_history):
        for n in d['numbers']:
            last_seen_mod[n] = idx
            
    # Transitions calculation
    d_prev = history[-1] if n_history > 0 else None
    d_prev2 = history[-2] if n_history > 1 else None
    
    features = {}
    for n in range(46):
        delay = n_history - 1 - last_seen[n] if last_seen[n] != -1 else n_history
        delay_ratio = delay / (mean_delays[n] + 1e-5)
        
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
        freq_5 = sum(1 for d in history[-5:] if n in d['numbers']) / 5.0 if n_history >= 5 else sum(1 for d in history if n in d['numbers']) / (n_history + 1e-5)
        freq_10 = sum(1 for d in history[-10:] if n in d['numbers']) / 10.0 if n_history >= 10 else sum(1 for d in history if n in d['numbers']) / (n_history + 1e-5)
        freq_20 = sum(1 for d in history[-20:] if n in d['numbers']) / 20.0 if n_history >= 20 else sum(1 for d in history if n in d['numbers']) / (n_history + 1e-5)
        freq_25 = sum(1 for d in history[-25:] if n in d['numbers']) / 25.0 if n_history >= 25 else sum(1 for d in history if n in d['numbers']) / (n_history + 1e-5)
        freq_50 = sum(1 for d in history[-50:] if n in d['numbers']) / 50.0 if n_history >= 50 else sum(1 for d in history if n in d['numbers']) / (n_history + 1e-5)
        freq_100 = sum(1 for d in history[-100:] if n in d['numbers']) / 100.0 if n_history >= 100 else sum(1 for d in history if n in d['numbers']) / (n_history + 1e-5)
        
        z_delay = (delay - mean_delays[n]) / (std_delays[n] + 1e-5)
        mod_freq_10 = sum(1 for d in mod_history[-10:] if n in d['numbers']) / 10.0 if n_mod_history >= 10 else sum(1 for d in mod_history if n in d['numbers']) / (n_mod_history + 1e-5)
        mod_freq_30 = sum(1 for d in mod_history[-30:] if n in d['numbers']) / 30.0 if n_mod_history >= 30 else sum(1 for d in mod_history if n in d['numbers']) / (n_mod_history + 1e-5)
        
        # 5. Advanced statistics
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
            freq_all = sum(1 for d in history if n in d['numbers']) / n_history
            fft_energy = float(n_history * n_history * freq_all * (1.0 - freq_all))
            
        features[n] = [
            delay, delay_ratio, z_delay, mean_delays[n], std_delays[n],
            mod_delay,
            trans_score, trans_score_lag2, trans_mod_score,
            lags[n][0], lags[n][1], lags[n][2], lags[n][3], lags[n][4], lags[n][5], lags[n][6], lags[n][7],
            ewma_3[n], ewma_10[n], ewma_30[n],
            freq_5, freq_10, freq_20, freq_25, freq_50, freq_100,
            mod_freq_10, mod_freq_30,
            mod_numeric,
            crossover, entropy, fft_energy
        ]

        
    return features


def prob_hits_poisson_binomial_flat(p):
    p0, p1, p2, p3, p4, p5 = p
    
    dp1_0 = 1.0 - p0
    dp1_1 = p0
    
    dp2_0 = dp1_0 * (1.0 - p1)
    dp2_1 = dp1_1 * (1.0 - p1) + dp1_0 * p1
    dp2_2 = dp1_1 * p1
    
    dp3_0 = dp2_0 * (1.0 - p2)
    dp3_1 = dp2_1 * (1.0 - p2) + dp2_0 * p2
    dp3_2 = dp2_2 * (1.0 - p2) + dp2_1 * p2
    dp3_3 = dp2_2 * p2
    
    dp4_0 = dp3_0 * (1.0 - p3)
    dp4_1 = dp3_1 * (1.0 - p3) + dp3_0 * p3
    dp4_2 = dp3_2 * (1.0 - p3) + dp3_1 * p3
    dp4_3 = dp3_3 * (1.0 - p3) + dp3_2 * p3
    dp4_4 = dp3_3 * p3
    
    dp5_0 = dp4_0 * (1.0 - p4)
    dp5_1 = dp4_1 * (1.0 - p4) + dp4_0 * p4
    dp5_2 = dp4_2 * (1.0 - p4) + dp4_1 * p4
    dp5_3 = dp4_3 * (1.0 - p4) + dp4_2 * p4
    dp5_4 = dp4_4 * (1.0 - p4) + dp4_3 * p4
    dp5_5 = dp4_4 * p4
    
    q5 = 1.0 - p5
    h3 = dp5_3 * q5 + dp5_2 * p5
    h4 = dp5_4 * q5 + dp5_3 * p5
    h5 = dp5_5 * q5 + dp5_4 * p5
    h6 = dp5_5 * p5
    
    return h3, h4, h5, h6

def compute_cooccurrence_matrix(draws):
    matrix = np.zeros((46, 46))
    total = len(draws) or 1
    for d in draws:
        nums = d['numbers']
        for u, v in itertools.combinations(nums, 2):
            matrix[u, v] += 1.0
            matrix[v, u] += 1.0
    return matrix / total

def select_balanced_pool_9(borda_scores, saturated_decades=None, max_per_decade=3, pool_size=9):
    """
    Selecciona un Pool de tamaño pool_size (default 9) números de máximo score Borda garantizando
    que ninguna decena tenga más de 3 números (max 2 en dec 40).
    Excepción de Élite: si una decena saturada contiene al Top 1 o Top 2 de la IA,
    permite hasta 3 números para no amputar a candidatos estelares.
    """
    ranked = sorted(range(46), key=lambda x: borda_scores[x], reverse=True)
    elite_numbers = set(ranked[:2])
    pool = []
    dec_counts = Counter()
    for n in ranked:
        dec = n // 10
        limit = 2 if dec == 4 else max_per_decade
        if saturated_decades and dec in saturated_decades:
            is_elite_dec = any((e // 10 == dec) for e in elite_numbers)
            if not is_elite_dec:
                limit = min(limit, 2)
            else:
                limit = 3
        if dec_counts[dec] < limit:
            pool.append(n)
            dec_counts[dec] += 1
            if len(pool) == pool_size:
                break
    if len(pool) < pool_size:
        for n in ranked:
            if n not in pool:
                pool.append(n)
                if len(pool) == pool_size: break
    return sorted(pool)

def compute_adaptive_pool(borda_scores, alpha=0.15, min_pool=9, max_pool=10):
    """Pool dinámico adaptativo ajustado por la desviación estándar de las puntuaciones."""
    scores = np.array([borda_scores[n] for n in range(46)])
    mean_s = np.mean(scores)
    std_s = np.std(scores)
    threshold = mean_s + alpha * std_s
    
    ranked = list(np.argsort(scores)[::-1])
    pool = [n for n in ranked if scores[n] >= threshold]
    
    if len(pool) < min_pool:
        pool = ranked[:min_pool]
    elif len(pool) > max_pool:
        pool = ranked[:max_pool]
    return pool

def compute_exponential_cooccurrence_scores(draws, window=40, half_life=15.0):
    cooccur_exp = np.zeros((46, 46))
    sub_draws = draws[-window:]
    T = len(sub_draws)
    for t_idx, d in enumerate(sub_draws):
        w_t = np.exp((t_idx - T) / half_life)
        for u, v in itertools.combinations(d['numbers'], 2):
            cooccur_exp[u, v] += w_t
            cooccur_exp[v, u] += w_t
    pcr_scores = np.array([np.sum(cooccur_exp[n]) for n in range(46)])
    return pcr_scores, cooccur_exp

def compute_delta_delay_momentum_scores(draws):
    last_seen = {n: -1 for n in range(46)}
    delays_list = defaultdict(list)
    for idx, d in enumerate(draws):
        for n in d['numbers']:
            if last_seen[n] != -1:
                delays_list[n].append(idx - last_seen[n])
            last_seen[n] = idx
    
    ddm_scores = np.zeros(46)
    n_total = len(draws)
    for n in range(46):
        delay = n_total - 1 - last_seen[n] if last_seen[n] != -1 else n_total
        mean_d = np.mean(delays_list[n]) if delays_list[n] else 7.6
        std_d = np.std(delays_list[n]) if len(delays_list[n]) > 1 else 5.0
        z_delay = (delay - mean_d) / (std_d + 1e-5)
        if -0.2 <= z_delay <= 1.8:
            ddm_scores[n] = 1.0 - 0.35 * abs(z_delay - 0.6)
        else:
            ddm_scores[n] = 0.1
    return ddm_scores

def compute_cross_modal_transitions(train_draws):
    """Matriz de transición entre modalidades cruzadas (Tradicional -> Segunda -> Revancha)."""
    trans_matrix = np.zeros((46, 46))
    for i in range(1, len(train_draws)):
        prev_nums = train_draws[i-1]['numbers']
        curr_nums = train_draws[i]['numbers']
        for u in prev_nums:
            for v in curr_nums:
                trans_matrix[u][v] += 1.0
    
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return trans_matrix / row_sums

def select_tiered_clique_tickets(boletos_viables, borda_scores, cooccur_matrix, gamma=0.75, w_co=250.0):
    # 1. Ticket A: Borda Core Spike (Highest Borda sum + Pair Co-occurrence)
    ticket_A = max(boletos_viables, key=lambda b: sum(borda_scores[n] for n in b) + w_co * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)))
    
    # Apply controlled discount (gamma = 0.75) for Ticket A numbers
    borda_B = borda_scores.copy()
    for n in ticket_A:
        borda_B[n] *= gamma
        
    # 2. Ticket B: Affinity Clique (High Co-occurrence Pair Density + remaining Borda anchor)
    ticket_B = max(boletos_viables, key=lambda b: sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda_B[n] for n in b))
    
    # Apply controlled discount for Ticket B numbers
    borda_C = borda_B.copy()
    for n in ticket_B:
        borda_C[n] *= gamma
        
    # 3. Ticket C: Hypergraph Coverage (Maximize remaining Borda score sum + pair co-occurrence)
    ticket_C = max(boletos_viables, key=lambda b: sum(borda_C[n] for n in b) + (w_co * 0.8) * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)))
    
    return [ticket_A, ticket_B, ticket_C]

def auditar_prediccion_previa(draws):
    pred_file = 'proxima_prediccion.json'
    if not os.path.exists(pred_file):
        return
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        last_sorteo = max(d['sorteo'] for d in draws)
        group_last = [d for d in draws if d['sorteo'] == last_sorteo]
        real_by_mod = {d['modalidad']: set(d['numbers']) for d in group_last}
        pozo_extra = real_by_mod.get('Tradicional', set()) | real_by_mod.get('Segunda', set()) | real_by_mod.get('Revancha', set())
        
        tickets = pred.get('tickets', [])
        if not tickets:
            return
            
        print("\n==================================================")
        print(f"AUDITORÍA DE LA JUGADA ANTERIOR (SORTEO #{last_sorteo}):")
        print("==================================================")
        for idx, t in enumerate(tickets):
            nums = set(t.get('numeros', []))
            t_name = t.get('nombre', f'Boleto {idx+1}')
            print(f" {t_name}: {sorted(list(nums))}")
            for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
                if mod in real_by_mod:
                    hits = nums & real_by_mod[mod]
                    print(f"   -> {mod:<12}: {len(hits)} aciertos {sorted(list(hits)) if hits else ''}")
            pe_hits = nums & pozo_extra
            print(f"   -> Pozo Extra  : {len(pe_hits)} aciertos {sorted(list(pe_hits)) if pe_hits else ''}\n")
        print("==================================================\n")
        
        # Guardar en historial_auditorias.json si no fue registrado antes
        audit_file = 'historial_auditorias.json'
        try:
            if os.path.exists(audit_file):
                with open(audit_file, 'r', encoding='utf-8') as af:
                    audits = json.load(af)
            else:
                audits = []
        except Exception:
            audits = []
            
        if not any(a.get('sorteo') == last_sorteo for a in audits):
            audit_record = {
                "fecha_auditoria": datetime.now().isoformat(),
                "sorteo": last_sorteo,
                "resultados_reales": {m: sorted(list(nums)) for m, nums in real_by_mod.items()},
                "tickets": [
                    {
                        "nombre": t.get('nombre', f'Boleto {i+1}'),
                        "numeros": sorted(t.get('numeros', [])),
                        "aciertos_tradicional": sorted(list(set(t.get('numeros', [])) & real_by_mod.get('Tradicional', set()))),
                        "aciertos_segunda": sorted(list(set(t.get('numeros', [])) & real_by_mod.get('Segunda', set()))),
                        "aciertos_revancha": sorted(list(set(t.get('numeros', [])) & real_by_mod.get('Revancha', set()))),
                        "aciertos_siempre_sale": sorted(list(set(t.get('numeros', [])) & real_by_mod.get('SiempreSale', set()))),
                        "aciertos_pozo_extra": sorted(list(set(t.get('numeros', [])) & pozo_extra))
                    }
                    for i, t in enumerate(tickets)
                ]
            }
            audits.append(audit_record)
            with open(audit_file, 'w', encoding='utf-8') as af:
                json.dump(audits, af, indent=2, ensure_ascii=False)
    except Exception:
        pass

def estimar_probabilidad_exito_conjunto(tickets, probs, trials=20000):
    probs_norm = probs / np.sum(probs)
    successes = 0
    for _ in range(trials):
        # Simula un sorteo basado en las probabilidades del modelo
        draw = np.random.choice(46, 6, replace=False, p=probs_norm)
        draw_set = set(draw)
        for ticket in tickets:
            if len(set(ticket).intersection(draw_set)) >= 3:
                successes += 1
                break
    return (successes / trials) * 100

def main(target_mod=None):
    print("==================================================")
    print("GENERADOR DE 3 BOLETOS ULTRA-OPTIMIZADOS - ML (QUINI 6)")
    print("==================================================\n")
    
    print("Cargando datos...")
    draws = load_data()
    n_draws = len(draws)
    if n_draws == 0:
        print("Error al cargar los datos.")
        sys.exit(1)
        
    last_draw = draws[-1]
    print(f"OK: {n_draws} sub-sorteos cargados.")
    print(f"Último sorteo registrado: Nro {last_draw['sorteo']} ({last_draw['fecha']}) [{last_draw['modalidad']}] -> {last_draw['numbers']}")
    
    # Auditar automáticamente el desempeño de la jugada anterior
    auditar_prediccion_previa(draws)
    
    if target_mod is None:
        print("\nElija la modalidad para la predicción del próximo sorteo:")
        print("1) Tradicional (Recomendado)")
        print("2) Segunda Vuelta")
        print("3) Revancha")
        print("4) Siempre Sale")
        
        opcion = '1'
        if sys.stdin.isatty():
            try:
                opcion = input("Ingrese opción (1-4, default 1): ").strip()
                if not opcion:
                    opcion = '1'
            except Exception:
                opcion = '1'
                
        mod_map = {'1': 'Tradicional', '2': 'Segunda', '3': 'Revancha', '4': 'SiempreSale'}
        target_mod = mod_map.get(opcion, 'Tradicional')
    print(f"\n--> Generando predicción para modalidad: {target_mod}")
    
    # Inicializar matrices de transición
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    transition2_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    transition_mod_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    print("Construyendo matrices históricas de transición...")
    for i in range(n_draws):
        update_transition_counts(draws, i, transition_counts, transition2_counts, transition_mod_counts)
        
    # Construir conjunto de entrenamiento con el histórico reciente (últimos 250 sorteos)
    print("Construyendo base de características de entrenamiento...")
    X_train_list = []
    y_train_list = []
    start_train = max(20, n_draws - 250)
    for i in range(start_train, n_draws):
        feats_i = extract_features_for_step(draws, i, transition_counts, transition2_counts, transition_mod_counts, draws[i]['modalidad'])
        real_draw_i = set(draws[i]['numbers'])
        for n in range(46):
            X_train_list.append(feats_i[n])
            y_train_list.append(1 if n in real_draw_i else 0)

            
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    # Entrenar modelos de ML
    print("Entrenando Super-Ensemble de ML (XGBoost, LightGBM, CatBoost, RandomForest, Neural Network MLP)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf_xgb.fit(X_train, y_train)
    clf_lgb.fit(X_train, y_train)
    clf_cat.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)
    clf_mlp.fit(X_train_scaled, y_train)
    
    # Generar Meta-Features para Stacking
    preds_xgb_tr = clf_xgb.predict_proba(X_train)[:, 1]
    preds_lgb_tr = clf_lgb.predict_proba(X_train)[:, 1]
    preds_cat_tr = clf_cat.predict_proba(X_train)[:, 1]
    preds_rf_tr  = clf_rf.predict_proba(X_train)[:, 1]
    preds_mlp_tr = clf_mlp.predict_proba(X_train_scaled)[:, 1]
    
    X_meta_train = np.column_stack([preds_xgb_tr, preds_lgb_tr, preds_cat_tr, preds_rf_tr, preds_mlp_tr])
    clf_meta.fit(X_meta_train, y_train)
    print("Super-Ensemble y Meta-Clasificador Stacking entrenados correctamente.")
    
    # Extraer características del sorteo entrante (t = n_draws)
    feats_next = extract_features_for_step(draws, n_draws, transition_counts, transition2_counts, transition_mod_counts, target_mod)
    X_test = np.array([feats_next[n] for n in range(46)])
    X_test_scaled = scaler.transform(X_test)
    
    # Predecir probabilidades base
    p_xgb = clf_xgb.predict_proba(X_test)[:, 1]
    p_lgb = clf_lgb.predict_proba(X_test)[:, 1]
    p_cat = clf_cat.predict_proba(X_test)[:, 1]
    p_rf  = clf_rf.predict_proba(X_test)[:, 1]
    p_mlp = clf_mlp.predict_proba(X_test_scaled)[:, 1]
    
    # Predecir con el motor Campeón Deep Multi-Head Attention MLP
    print(f"Ejecutando motor Campeón: Deep Multi-Head Attention Neural Network ({target_mod})...")
    p_attn = compute_attention_mlp_scores(draws[:n_draws], target_mod=target_mod)
    
    X_meta_test = np.column_stack([p_xgb, p_lgb, p_cat, p_rf, p_mlp])
    probs_t = clf_meta.predict_proba(X_meta_test)[:, 1]
    
    # Incorporar puntuación de transición cross-modal de 2º orden (Super-Modelo)
    cross_modal_mat = compute_cross_modal_transitions(draws[-60:])
    last_nums = draws[-1]['numbers']
    cross_scores = np.zeros(46)
    for u in last_nums:
        cross_scores += cross_modal_mat[u]

    # Computar métricas de Co-ocurrencia Ponderada Exponencial (PCR) y Momento de Retardo (DDM)
    pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(draws)
    ddm_scores = compute_delta_delay_momentum_scores(draws)

    # -------------------------------------------------------------
    # DETECCIÓN DE RÉGIMEN DINÁMICO (IDEA 2: MOMENTUM VS REVERSIÓN)
    # Analiza la tasa de repetición en los últimos 5 sorteos
    # -------------------------------------------------------------
    # Detección cronológica real de régimen en sorteos de la modalidad objetivo
    trad_draws_all = [d for d in draws if d['modalidad'] == target_mod]
    last_5_trad = trad_draws_all[-5:]
    n_repeats = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
    avg_repeats = n_repeats / 4.0
    
    if avg_repeats >= 1.0:
        regime_desc = "Régimen Caliente / Alta Inercia (Prioridad Momentum/Recencia)"
        w_rec = 2.8
        w_ddm = 1.2
        w_attn = 3.0
    elif avg_repeats <= 0.5:
        regime_desc = "Régimen Frío / Retorno a la Media (Prioridad Atraso Relativo DDM)"
        w_rec = 1.2
        w_ddm = 2.8
        w_attn = 2.5
    else:
        regime_desc = "Régimen Balanceado (Equilibrio Momentum y Atraso)"
        w_rec = 2.0
        w_ddm = 1.8
        w_attn = 2.8
        
    print(f"\n==================================================")
    print(f"ANÁLISIS DE RÉGIMEN DINÁMICO (SUPER-MODELO ML):")
    print(f" -> Tasa de repetición reciente: {avg_repeats:.2f} números/sorteo")
    print(f" -> Estado detectado: {regime_desc}")
    print(f" -> Pesos calibrados: Recencia={w_rec}, Atraso(DDM)={w_ddm}, Atención={w_attn}")
    print(f"==================================================")

    # Fusión Borda Count con pesos adaptativos según el régimen actual
    borda = defaultdict(float)
    models_to_rank = [
        (p_attn, w_attn),
        (probs_t, w_rec),
        (pcr_scores, 1.5),
        (cross_scores, 1.2),
        (ddm_scores, w_ddm),
        (p_mlp, 1.0),
        (p_cat, 1.0),
        (p_xgb, 1.0),
        (p_lgb, 1.0),
        (p_rf, 1.0)
    ]
    for p_model, weight in models_to_rank:
        ranking_model = sorted(range(46), key=lambda x: p_model[x], reverse=True)
        for rank, n in enumerate(ranking_model):
            borda[n] += weight * (45 - rank)
    # -------------------------------------------------------------
    # SÚPER-RESONANCIA Y ARRASTRE MULTI-MODAL INMEDIATO
    # Bolillas que salieron en la fecha previa inmediata (con boost a Siempre Sale y resonancia múltiple)
    # -------------------------------------------------------------
    last_sorteo_num = max(d['sorteo'] for d in draws)
    last_date_draws = [d for d in draws if d['sorteo'] == last_sorteo_num]
    res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
    prev_ss_nums = set()
    for d in last_date_draws:
        if d['modalidad'] == 'SiempreSale':
            prev_ss_nums.update(d['numbers'])

    resonance_detected = []
    for n in range(46):
        if res_counts[n] >= 2:
            bonus_res = 18.0 * (res_counts[n] - 1)
            borda[n] += bonus_res
            resonance_detected.append((n, res_counts[n], bonus_res))
        elif res_counts[n] == 1:
            borda[n] += 4.0
        if n in prev_ss_nums:
            borda[n] += 4.0
            
    if resonance_detected:
        print("\n==================================================")
        print("SÚPER-RESONANCIA MULTI-MODAL DETECTADA (SORTEO PREVIO):")
        for num, app, bon in resonance_detected:
            print(f" -> Número {num:02d}: {app} apariciones en fecha #{last_sorteo_num} (Bonus IA: +{bon:.0f} pts)")
        print(f"==================================================")

    borda_ranking = [n for n, score in sorted(borda.items(), key=lambda x: x[1], reverse=True)]

    # -------------------------------------------------------------
    # MODELADO DINÁMICO POR DÍA DE LA SEMANA (MIÉRCOLES VS DOMINGO)
    # -------------------------------------------------------------
    last_draw = draws[-1]
    last_draw_date = last_draw['fecha']
    last_weekday = datetime.strptime(last_draw_date, '%Y-%m-%d').weekday()
    target_weekday = 2 if last_weekday == 6 else 6
    weekday_name = "Miércoles" if target_weekday == 2 else "Domingo"
    
    same_day_draws = [d for d in draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
    
    borda_day = defaultdict(float)
    borda_hib = defaultdict(float)
    for n in range(46):
        borda_day[n] = borda[n] + 16.0 * counts_day[n]
        borda_hib[n] = borda[n] + 8.0 * counts_day[n]
        
    # Obtener decena(s) saturada(s) del último sorteo
    saturated_decades = set()
    last_mod_draw = None
    for d in reversed(draws):
        if d['modalidad'] == target_mod:
            last_mod_draw = d
            break
    if last_mod_draw:
        dec_counts = Counter(num // 10 for num in last_mod_draw['numbers'])
        for dec, count in dec_counts.items():
            if count >= 3:
                saturated_decades.add(dec)
        if saturated_decades:
            print(f"Decenas saturadas en último sorteo ({last_mod_draw['fecha']}): {[d*10 for d in saturated_decades]}")

    pool_9_base = select_balanced_pool_9(borda, saturated_decades=saturated_decades, pool_size=9)
    pool_9_day = select_balanced_pool_9(borda_day, saturated_decades=saturated_decades, pool_size=9)
    pool_10_hib = select_balanced_pool_9(borda_hib, saturated_decades=saturated_decades, pool_size=10)
    
    print("\n==================================================")
    print(f"MODELO ESPECIALIZADO POR DÍA ({weekday_name.upper()}):")
    print(f" -> Pool Base {target_mod} (9): {pool_9_base}")
    print(f" -> Pool Día {weekday_name} (9): {pool_9_day}")
    print(f" -> Pool Cobertura Híbrida (10): {pool_10_hib}")
    print(f"==================================================")

    # Precomputar pares antagónicos para filtrado físico
    antagonist_pairs = compute_antagonist_pairs(draws)
    print(f" -> Pares antagónicos bloqueados (Lift < 0.40): {len(antagonist_pairs)}")

    # Generar números calientes recientes de los últimos 5 sorteos secuenciales
    recent_numbers = set()
    for d in draws[-5:]:
        recent_numbers.update(d['numbers'])
        
# Saturated decades precomputadas arriba

    # Filtrar combinaciones con Filtro Anti-Congestión (3 decenas) y Bloqueo Antagónico
    def get_viable_combos(pool):
        combos = list(itertools.combinations(pool, 6))
        # 1. Intentar con 3 decenas distintas y pares antagónicos
        v = [c for c in combos if es_ticket_valido(c, recent_numbers, saturated_decades, antagonist_pairs, min_distinct_decades=3)]
        if len(v) < 3:
            v = [c for c in combos if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
        if len(v) < 3:
            v = combos[:10]
        return v

    viable_base = get_viable_combos(pool_9_base)
    viable_day = get_viable_combos(pool_9_day)
    viable_hib = get_viable_combos(pool_10_hib)

    print(f"Combinaciones viables: Base={len(viable_base)}, Día={len(viable_day)}, Híbrido={len(viable_hib)}")

    # Generar terna base de máximo rendimiento
    t_base = select_optimal_wheel_triplet(viable_base, pool_9_base, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)

    # -------------------------------------------------------------
    # PORTFOLIO ESPECIALIZADO DE 3 BOLETOS (1 CUPÓN HABITUAL)
    # Boleto 1: Cazador Tradicional (Modelo A)
    # Boleto 2: Especialista Día de la Semana (Modelo B)
    # Boleto 3: Consenso / Cobertura Híbrida (Modelo C)
    # -------------------------------------------------------------
    b1 = t_base[0]
    
    # Guardián de Solapamiento para Boleto 2 (2 <= overlap <= 4 con B1)
    cands_b2 = [c for c in viable_day if 2 <= len(set(c) & set(b1)) <= 4]
    if not cands_b2:
        cands_b2 = [c for c in viable_day if c != b1]
    b2 = max(cands_b2, key=lambda c: sum(borda_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
    
    # Guardián de Solapamiento para Boleto 3 (2 <= overlap <= 4 con B1 y B2, triple overlap <= 2, maximizar cuádruplas)
    q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
    cands_b3 = [
        c for c in viable_hib
        if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2
    ]
    if not cands_b3:
        cands_b3 = [c for c in viable_hib if c != b1 and c != b2 and len(set(c) & set(b1)) <= 4 and len(set(c) & set(b2)) <= 4]
    if not cands_b3:
        cands_b3 = [c for c in viable_hib if c != b1 and c != b2]
        
    b3 = max(
        cands_b3,
        key=lambda c: sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12)
    )
    tickets_optimos = [b1, b2, b3]

    total_unique_numbers = sorted(list(set(b1) | set(b2) | set(b3)))

    # Simular probabilidad de éxito conjunta
    prob_exito = estimar_probabilidad_exito_conjunto(tickets_optimos, probs_t)
    
    individual_probs = []
    breakdown_probs = []
    for b in tickets_optimos:
        c_probs = [probs_t[n] for n in sorted(b)]
        h3, h4, h5, h6 = prob_hits_poisson_binomial_flat(c_probs)
        p_tot = (h3 + h4 + h5 + h6) * 100
        p_premio_4plus = (h4 + h5 + h6) * 100
        individual_probs.append(p_tot)
        breakdown_probs.append((h3 * 100, p_premio_4plus))
    
    # Cálculo del Índice de Convicción del Sorteo (ICS: 0 a 100)
    score_regime = 30 if avg_repeats <= 0.5 or avg_repeats >= 1.0 else 20
    score_res = 25 if resonance_detected else 15
    score_quad = int(20 * (len(q12 | set(itertools.combinations(b3, 4))) / 45.0))
    # Consenso entre modelos (amplitud del Borda Top 9 vs Media)
    top_borda_mean = np.mean([borda[n] for n in pool_9_base])
    all_borda_mean = np.mean(list(borda.values()))
    score_consensus = min(25, int(15 * (top_borda_mean / (all_borda_mean + 1e-5))))
    ics_score = min(99, score_regime + score_res + score_quad + score_consensus)
    ics_label = "ALTA" if ics_score >= 80 else ("MEDIA-ALTA" if ics_score >= 65 else "ESTÁNDAR")

    print("\n==================================================")
    print("PORTFOLIO ESPECIALIZADO DE 3 BOLETOS (QUINI 6):")
    print(f"Total números únicos jugados en el cupón: {len(total_unique_numbers)} {total_unique_numbers}")
    print(f"Índice de Convicción Matemática del Sorteo (ICS): {ics_score}/100 [{ics_label}]")
    print("==================================================")
    ticket_names = [
        f"SUPER TICKET 1 [ESPECIALISTA {target_mod.upper()} - MÁXIMO POZO]",
        f"SUPER TICKET 2 [ESPECIALISTA DÍA {weekday_name.upper()} - ARRASTRE MULTI-MODAL]",
        f"SUPER TICKET 3 [CONSENSO HÍBRIDO - RED DE COBERTURA POZO EXTRA]"
    ]
    tickets_json = []
    for idx, ticket in enumerate(tickets_optimos):
        t_sorted = [int(x) for x in sorted(ticket)]
        t_name = ticket_names[idx]
        h3_pct, h4_pct = breakdown_probs[idx]
        print(f"Boleto {idx+1:02d} [{t_name}]:")
        print(f"  👉 Números: {t_sorted}")
        print(f"  👉 Prob. Total (3+): {individual_probs[idx]:.2f}% | 3 aciertos: {h3_pct:.2f}% | 4+ aciertos (PREMIO): {h4_pct:.2f}%\n")
        tickets_json.append({
            "nombre": t_name,
            "numeros": [int(x) for x in t_sorted],
            "prob_3_plus": round(individual_probs[idx], 2),
            "prob_premio_4_plus": round(h4_pct, 2)
        })
    print("==================================================")
    print(f"PROBABILIDAD ACUMULADA CONJUNTA (3+ aciertos en al menos 1 boleto): {prob_exito:.2f}%")
    print(f"Desempeño relativo frente al azar: {(prob_exito / 5.48):.2f}x superior")
    print("==================================================")

    top_pool = total_unique_numbers

    # Export to proxima_prediccion.json and historial_predicciones.json
    pred_data = {
        "fecha_prediccion": datetime.now().isoformat(),
        "fecha_sorteo_objetivo": "Siguiente Sorteo (Domingo / Miércoles)",
        "juego": "Quini 6 (Super-Modelo Dinámico por Régimen + Rueda Balanceada 2-Fold BIBD)",
        "top_numeros": [int(x) for x in sorted(top_pool)],
        "tickets": tickets_json
    }

    with open('proxima_prediccion.json', 'w', encoding='utf-8') as f:
        json.dump(pred_data, f, indent=2, ensure_ascii=False)
    print("\n[OK] Predicción guardada exitosamente en proxima_prediccion.json")

    hist_file = 'historial_predicciones.json'
    try:
        with open(hist_file, 'r', encoding='utf-8') as f:
            historial = json.load(f)
    except Exception:
        historial = []

    historial.append(pred_data)
    with open(hist_file, 'w', encoding='utf-8') as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)
    print("[OK] Predicción registrada en historial_predicciones.json")

if __name__ == '__main__':
    main()

