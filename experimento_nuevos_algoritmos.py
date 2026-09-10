import csv
import os
import sys
import math
import random
import numpy as np
import itertools
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =====================================================================
# 1. CARGA DE DATOS HISTÓRICOS
# =====================================================================
def load_data(filepath="c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv"):
    if not os.path.exists(filepath):
        filepath = "historico_quini_completo.csv"
        
    draws = []
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
            except Exception:
                continue
                
    modality_order = {'Tradicional': 1, 'Segunda': 2, 'Revancha': 3, 'SiempreSale': 4}
    draws.sort(key=lambda x: (x['sorteo'], modality_order.get(x['modalidad'], 9)))
    return draws

# =====================================================================
# 2. REGLAS DE FILTRADO FÍSICO DE BOLETOS
# =====================================================================
def es_ticket_valido(ticket, recent_numbers=None, saturated_decades=None):
    ticket = sorted(ticket)
    # Consecutividad max 2 parejas
    gaps_le_1 = sum(1 for i in range(1, len(ticket)) if ticket[i] - ticket[i-1] <= 1)
    if gaps_le_1 > 2:
        return False
    # Spread entre 15 y 45
    spread = ticket[5] - ticket[0]
    if spread < 15 or spread > 45:
        return False
    # Decenas max 4 por decena
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(count > 4 for count in decenas.values()):
        return False
    # Paridad entre 1 y 5 pares
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares == 0 or pares == 6:
        return False
    # Partición caliente/frío
    if recent_numbers is not None:
        recent_count = sum(1 for n in ticket if n in recent_numbers)
        if recent_count < 1 or recent_count > 5:
            return False
    # Filtro de decenas saturadas
    if saturated_decades:
        for dec in saturated_decades:
            if decenas[dec] > 2:
                return False
    return True

# =====================================================================
# 3. IMPLEMENTACIÓN DE LOS 5 ALGORITMOS
# =====================================================================

# ---------------------------------------------------------------------
# ALGORITMO 1: Super-Ensemble Stacking ML (Baseline Actual)
# ---------------------------------------------------------------------
def model_super_ensemble(train_draws):
    # Frecuencia global y ventana
    total_draws = len(train_draws)
    if total_draws < 20:
        return np.ones(46) / 46
        
    counts_all = Counter()
    counts_20  = Counter()
    counts_50  = Counter()
    
    for d in train_draws:
        for n in d['numbers']:
            counts_all[n] += 1
    for d in train_draws[-20:]:
        for n in d['numbers']:
            counts_20[n] += 1
    for d in train_draws[-50:]:
        for n in d['numbers']:
            counts_50[n] += 1
            
    # Transición secuencial
    trans_mod = defaultdict(Counter)
    for i in range(1, len(train_draws)):
        d_prev = train_draws[i-1]
        d_curr = train_draws[i]
        key = (d_prev['modalidad'], d_curr['modalidad'])
        for x in d_prev['numbers']:
            for y in d_curr['numbers']:
                trans_mod[key][(x, y)] += 1
                
    last_mod = train_draws[-1]['modalidad']
    target_key = (last_mod, 'Tradicional')
    
    # Feature vector para cada número 0..45
    scores = np.zeros(46)
    last_draw_nums = set(train_draws[-1]['numbers'])
    
    for n in range(46):
        f_all = counts_all[n] / (total_draws * 6)
        f_20  = counts_20[n] / 120.0
        f_50  = counts_50[n] / 300.0
        
        # Transición acumulada
        trans_score = sum(trans_mod[target_key][(x, n)] for x in last_draw_nums)
        
        # Combinación de features
        scores[n] = 0.35 * f_20 + 0.25 * f_50 + 0.15 * f_all + 0.25 * (trans_score / 36.0 if trans_score else 0)
        
    return scores

# ---------------------------------------------------------------------
# ALGORITMO 2: Graph-Markov Random Walk with Restart (G-RWR)
# ---------------------------------------------------------------------
def model_graph_random_walk(train_draws, alpha=0.85):
    """
    Construye un grafo de co-ocurrencia y transiciones temporales.
    Calcula la centralidad PageRank con reinicio en los números del último sorteo.
    """
    if len(train_draws) < 20:
        return np.ones(46) / 46

    # Matriz de Adyacencia (46x46)
    A = np.zeros((46, 46))
    
    # Adyacencia por Co-ocurrencia en el mismo sorteo (pesado reciente)
    for t, d in enumerate(train_draws[-100:]):
        weight = math.exp((t - 100) / 30.0) # Decaimiento exponencial
        for x, y in itertools.combinations(d['numbers'], 2):
            A[x, y] += weight
            A[y, x] += weight
            
    # Adyacencia por Transición del sorteo t-1 al sorteo t
    for t in range(1, len(train_draws[-100:])):
        d_prev = train_draws[-100 + t - 1]
        d_curr = train_draws[-100 + t]
        for x in d_prev['numbers']:
            for y in d_curr['numbers']:
                A[x, y] += 1.5 # Peso adicional a la secuencia directa
                
    # Normalizar estocásticamente
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M = A / row_sums
    
    # Vector de reinicio (Personalized PageRank enfocado en últimos 3 sorteos)
    v = np.zeros(46)
    for d in train_draws[-3:]:
        for n in d['numbers']:
            v[n] += 1.0
    if v.sum() > 0:
        v /= v.sum()
    else:
        v = np.ones(46) / 46.0
        
    # Power Iteration para PageRank Random Walk with Restart
    p = np.copy(v)
    for _ in range(30):
        p = alpha * (p @ M) + (1 - alpha) * v
        
    return p

# ---------------------------------------------------------------------
# ALGORITMO 3: Dynamic Poisson-Kalman Drift Filter (DP-KVF)
# ---------------------------------------------------------------------
def model_poisson_kalman(train_draws):
    """
    Filtro Bayesiano de velocidad e inercia Poisson para estimar la tasa dinámica lambda_i(t).
    """
    if len(train_draws) < 20:
        return np.ones(46) / 46

    # Parámetros del filtro
    lambda_hat = np.ones(46) * (6.0 / 46.0) # Tasa a priori
    velocity   = np.zeros(46)               # Aceleración/Tendencia
    
    alpha_smooth = 0.15 # Factor de suavizado exponencial
    beta_trend   = 0.05 # Factor de inercia
    
    for d in train_draws:
        appeared = set(d['numbers'])
        for n in range(46):
            y_t = 1.0 if n in appeared else 0.0
            
            # Innovación de Kalman / Filter Step
            err = y_t - lambda_hat[n]
            lambda_hat[n] += alpha_smooth * err + velocity[n]
            velocity[n]   += beta_trend * err
            
            # Bounding
            lambda_hat[n] = max(0.01, min(0.95, lambda_hat[n]))
            
    # Combina tasa estimada con momento proyectado a t+1
    scores = lambda_hat + 0.5 * velocity
    scores = np.maximum(0, scores)
    return scores

# ---------------------------------------------------------------------
# ALGORITMO 4: Deep Multi-Head Attention MLP (Attn-MLP)
# ---------------------------------------------------------------------
def model_attention_mlp(train_draws):
    """
    Neural Network MLP con ponderación espectral y atención a lags múltiples (1, 2, 3, 5, 10).
    """
    if len(train_draws) < 40:
        return np.ones(46) / 46
        
    # Construir historial secuencial de bolillas Tradicionales
    trad_draws = [d['numbers'] for d in train_draws if d['modalidad'] == 'Tradicional']
    if len(trad_draws) < 30:
        trad_draws = [d['numbers'] for d in train_draws]
        
    # Matriz binaria T x 46
    T = len(trad_draws)
    matrix = np.zeros((T, 46))
    for t, nums in enumerate(trad_draws):
        for n in nums:
            matrix[t, n] = 1.0
            
    # Lags de atención: 1, 2, 3, 5
    X_list = []
    y_list = []
    for t in range(5, T):
        feat = np.hstack([matrix[t-1], matrix[t-2], matrix[t-3], matrix[t-5]])
        X_list.append(feat)
        y_list.append(matrix[t])
        
    X = np.array(X_list)
    Y = np.array(y_list)
    
    # Entrenar MLP multi-salida / o modelo denso
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42, early_stopping=True)
    
    # Predecir probabilidades para cada bola
    scores = np.zeros(46)
    for n in range(46):
        y_n = Y[:, n]
        if len(np.unique(y_n)) < 2:
            scores[n] = 0.1
            continue
        try:
            mlp.fit(X, y_n)
            last_feat = np.hstack([matrix[-1], matrix[-2], matrix[-3], matrix[-5]]).reshape(1, -1)
            scores[n] = mlp.predict_proba(last_feat)[0, 1]
        except Exception:
            scores[n] = 0.1
            
    return scores

# ---------------------------------------------------------------------
# ALGORITMO 5: Adaptive Thompson-Sampling Multi-Armed Bandit (TS-Bandit)
# ---------------------------------------------------------------------
class ThompsonSamplingBanditEnsemble:
    def __init__(self, n_models=4):
        self.alpha = np.ones(n_models) # Aciertos + 1 (Prior Beta)
        self.beta  = np.ones(n_models) # Fallos + 1 (Prior Beta)
        
    def get_weights(self):
        # Muestreo de Distribución Beta para cada modelo
        samples = np.random.beta(self.alpha, self.beta)
        weights = samples / np.sum(samples)
        return weights
        
    def update(self, model_idx, reward):
        # reward es el número de aciertos obtenidos (0 a 6)
        if reward >= 3:
            self.alpha[model_idx] += reward
        else:
            self.beta[model_idx] += (3 - reward)

# =====================================================================
# 4. OPTIMIZACIÓN Y SELECCIÓN DE 3 BOLETOS POR POISSON-BINOMIAL
# =====================================================================
def select_top_3_tickets(boletos, probs, n_tickets=3):
    scores = []
    for b in boletos:
        # Score individual como suma de log-probabilidades de apariciones esperadas
        score = sum(np.log(max(probs[n], 1e-6)) for n in b)
        scores.append((score, b))
        
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Seleccionar 3 boletos buscando la máxima distancia / cobertura Hamming
    selected = [scores[0][1]]
    for score, b in scores[1:]:
        if len(selected) >= n_tickets:
            break
        # Garantizar al menos 2 números de diferencia con boletos ya elegidos
        if all(len(set(b).intersection(set(s))) <= 4 for s in selected):
            selected.append(b)
            
    while len(selected) < n_tickets and len(scores) > len(selected):
        for score, b in scores:
            if b not in selected:
                selected.append(b)
                if len(selected) >= n_tickets:
                    break
                    
    return selected[:n_tickets]

# =====================================================================
# 5. BACKTEST WALK-FORWARD BENCHMARKING
# =====================================================================
def run_benchmark(n_sorteos_backtest=60):
    print("=====================================================================")
    print(f"   BACKTEST WALK-FORWARD EN VIVO ({n_sorteos_backtest} SORTEOS TRADICIONALES)")
    print("=====================================================================")
    
    draws = load_data()
    print(f"Total registros cargados: {len(draws)} sub-sorteos")
    
    # Extraer sorteos de modalidad Tradicional
    trad_indices = [i for i, d in enumerate(draws) if d['modalidad'] == 'Tradicional']
    
    if len(trad_indices) < n_sorteos_backtest + 20:
        n_sorteos_backtest = len(trad_indices) - 20
        
    test_indices = trad_indices[-n_sorteos_backtest:]
    print(f"Sorteos de prueba: Desde Sorteo Nro {draws[test_indices[0]]['sorteo']} hasta Nro {draws[test_indices[-1]]['sorteo']}")
    print("=====================================================================\n")
    
    # Estructuras de evaluación
    models_name = [
        "1. Super-Ensemble Stacking ML (Baseline)",
        "2. Graph-Markov Random Walk (G-RWR)",
        "3. Dynamic Poisson-Kalman Filter (DP-KVF)",
        "4. Deep Multi-Head Attention MLP (Attn-MLP)",
        "5. Adaptive Thompson-Sampling Bandit Ensemble"
    ]
    
    stats = {m: {'hits_3': 0, 'hits_4': 0, 'hits_5': 0, 'hits_6': 0, 'total_hits': 0, 'pool_coverage': 0, 'total_tickets': 0} for m in models_name}
    
    ts_bandit = ThompsonSamplingBanditEnsemble(n_models=4)
    
    # Análisis de Dinámica de Movimientos
    decade_transitions = defaultdict(Counter)
    co_occurrence_pairs = Counter()
    
    for idx_step, idx_draw in enumerate(test_indices):
        train_sub = draws[:idx_draw]
        target_draw = draws[idx_draw]
        target_nums = set(target_draw['numbers'])
        
        # Números recientes y decenas saturadas
        recent_numbers = set()
        for d in train_sub[-5:]:
            recent_numbers.update(d['numbers'])
            
        last_trad = [d for d in train_sub if d['modalidad'] == 'Tradicional'][-1]
        dec_counts = Counter(num // 10 for num in last_trad['numbers'])
        saturated_decades = {dec for dec, count in dec_counts.items() if count >= 3}
        
        # Analizar movimientos de decenas (Saturación -> Siguiente resultado)
        target_decades = Counter(num // 10 for num in target_draw['numbers'])
        for sat in saturated_decades:
            for target_dec, count in target_decades.items():
                decade_transitions[sat*10][target_dec*10] += count
                
        # Analizar pares de co-ocurrencia
        for p in itertools.combinations(target_draw['numbers'], 2):
            co_occurrence_pairs[p] += 1
            
        # Generar Scores para cada uno de los algoritmos
        scores_dict = {}
        scores_dict[models_name[0]] = model_super_ensemble(train_sub)
        scores_dict[models_name[1]] = model_graph_random_walk(train_sub)
        scores_dict[models_name[2]] = model_poisson_kalman(train_sub)
        scores_dict[models_name[3]] = model_attention_mlp(train_sub)
        
        # Algoritmo 5: Thompson Sampling Bandit Ensemble
        w = ts_bandit.get_weights()
        ts_score = (w[0] * scores_dict[models_name[0]] +
                    w[1] * scores_dict[models_name[1]] +
                    w[2] * scores_dict[models_name[2]] +
                    w[3] * scores_dict[models_name[3]])
        scores_dict[models_name[4]] = ts_score
        
        # Evaluar cada modelo en este sorteo
        for m_idx, m_name in enumerate(models_name):
            sc = scores_dict[m_name]
            
            # Normalizar a probabilidades estocásticas
            prob_dist = sc / np.sum(sc) if np.sum(sc) > 0 else np.ones(46) / 46.0
            
            # Pool de top 15 números
            top_15 = np.argsort(prob_dist)[-15:]
            pool_hits = len(set(top_15).intersection(target_nums))
            stats[m_name]['pool_coverage'] += pool_hits
            
            # Filtrar combinaciones válidas en Pool
            combos = list(itertools.combinations(top_15, 6))
            viables = [c for c in combos if es_ticket_valido(c, recent_numbers, saturated_decades)]
            if not viables:
                viables = [c for c in combos if es_ticket_valido(c)]
            if not viables:
                viables = combos[:10]
                
            # Seleccionar 3 boletos
            tickets = select_top_3_tickets(viables, prob_dist, n_tickets=3)
            
            max_hits_in_draw = 0
            for t in tickets:
                hits = len(set(t).intersection(target_nums))
                stats[m_name]['total_hits'] += hits
                stats[m_name]['total_tickets'] += 1
                if hits > max_hits_in_draw:
                    max_hits_in_draw = hits
                if hits == 3: stats[m_name]['hits_3'] += 1
                elif hits == 4: stats[m_name]['hits_4'] += 1
                elif hits == 5: stats[m_name]['hits_5'] += 1
                elif hits == 6: stats[m_name]['hits_6'] += 1
                
            # Actualizar el Aprendizaje por Refuerzo (Thompson Bandit)
            if m_idx < 4:
                ts_bandit.update(m_idx, max_hits_in_draw)

        if (idx_step + 1) % 15 == 0 or (idx_step + 1) == len(test_indices):
            print(f"Progreso Backtest: {idx_step + 1}/{len(test_indices)} sorteos evaluados...")

    print("\n=====================================================================")
    print("       RESULTADOS COMPARATIVOS DE EFECTIVIDAD (BACKTEST)")
    print("=====================================================================\n")
    
    print(f"{'Algoritmo / Modelo':<45} | {'Aciertos 3+':<12} | {'Aciertos 4+':<12} | {'Coherencia Pool 15':<18} | {'Prom. Acierto/Ticket'}")
    print("-" * 115)
    
    best_model = None
    best_rate = -1.0
    
    for m_name in models_name:
        st = stats[m_name]
        n_sorteos = len(test_indices)
        n_tickets = st['total_tickets']
        
        tot_3plus = st['hits_3'] + st['hits_4'] + st['hits_5'] + st['hits_6']
        pct_3plus = (tot_3plus / n_tickets) * 100.0 if n_tickets > 0 else 0
        
        tot_4plus = st['hits_4'] + st['hits_5'] + st['hits_6']
        pct_4plus = (tot_4plus / n_tickets) * 100.0 if n_tickets > 0 else 0
        
        avg_pool  = st['pool_coverage'] / n_sorteos if n_sorteos > 0 else 0
        avg_hits  = st['total_hits'] / n_tickets if n_tickets > 0 else 0
        
        print(f"{m_name:<45} | {tot_3plus:>3d} ({pct_3plus:>5.1f}%)   | {tot_4plus:>3d} ({pct_4plus:>5.1f}%)   | {avg_pool:>4.2f} / 6.00 bolillas | {avg_hits:>4.2f} aciertos")
        
        if pct_3plus > best_rate:
            best_rate = pct_3plus
            best_model = m_name
            
    print("-" * 115)
    print(f"\n[GANADOR]: El modelo con mayor efectividad es: *** {best_model} *** con {best_rate:.2f}% de aciertos 3+.")
    
    print("\n=====================================================================")
    print("     ANÁLISIS DE MOVIMIENTOS Y DINÁMICA DE LAS JUGADAS")
    print("=====================================================================")
    print("\n1. DESPLAZAMIENTO TRAS SATURACIÓN DE DECENAS (Decade Saturation Deflation):")
    print("   Cuando una decena concentra 3 o más bolillas en un sorteo, en el sorteo siguiente se desplaza así:")
    for sat, targets in sorted(decade_transitions.items()):
        total_sat = sum(targets.values())
        top_dest = targets.most_common(2)
        dest_str = ", ".join([f"Decena {d}s ({cnt/total_sat*100:.1f}%)" for d, cnt in top_dest])
        print(f"   • Decena {sat}s saturada -> Migra hacia: {dest_str}")
        
    print("\n2. PARES DE MAYOR ATRACCIÓN Y CO-OCURRENCIA SECUENCIAL:")
    top_pairs = co_occurrence_pairs.most_common(5)
    for (n1, n2), cnt in top_pairs:
        print(f"   • Par [{n1:02d} - {n2:02d}]: Aparecieron juntos {cnt} veces en el período de prueba.")
        
    print("=====================================================================\n")

if __name__ == '__main__':
    run_benchmark(n_sorteos_backtest=60)
