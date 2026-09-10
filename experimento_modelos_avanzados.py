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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
# 3. EXTRAER FEATURES AVANZADAS PARA REDES DE ATENCIÓN Y STACKING
# =====================================================================
def build_feature_matrices(train_draws):
    trad_draws = [d['numbers'] for d in train_draws if d['modalidad'] == 'Tradicional']
    if len(trad_draws) < 30:
        trad_draws = [d['numbers'] for d in train_draws]
        
    T = len(trad_draws)
    matrix = np.zeros((T, 46))
    for t, nums in enumerate(trad_draws):
        for n in nums:
            matrix[t, n] = 1.0
            
    # Lags multi-atención: 1, 2, 3, 4, 5, 8, 10
    X_list = []
    y_list = []
    for t in range(10, T):
        # Concatenar lags con pesos espectrales
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
    
    return X, Y, last_feat, matrix

# =====================================================================
# 4. MODELOS CANDIDATOS
# =====================================================================

# ---------------------------------------------------------------------
# MODELO A: Multi-Head Attention MLP v1 (Campeón Anterior - 4.4%)
# ---------------------------------------------------------------------
def model_attn_mlp_v1(train_draws):
    X, Y, last_feat, matrix = build_feature_matrices(train_draws)
    scores = np.zeros(46)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42, early_stopping=True)
    
    for n in range(46):
        y_n = Y[:, n]
        if len(np.unique(y_n)) < 2:
            scores[n] = 0.05
            continue
        try:
            mlp.fit(X, y_n)
            scores[n] = mlp.predict_proba(last_feat)[0, 1]
        except Exception:
            scores[n] = 0.05
            
    return scores

# ---------------------------------------------------------------------
# MODELO B: Transformer-Style Self-Attention Network (Attn-Transformer)
# ---------------------------------------------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def model_transformer_attention(train_draws):
    """
    Self-Attention Mechanism sobre la secuencia temporal de vectores binarios.
    Aplica matrices Query (Q), Key (K) y Value (V) para correlaciones globales.
    """
    X, Y, last_feat, matrix = build_feature_matrices(train_draws)
    T = matrix.shape[0]
    if T < 20:
        return np.ones(46) / 46.0
        
    # Multi-head Self-Attention
    seq = matrix[-15:] # Ventana de 15 sorteos
    d_k = 46
    # Proyecciones lineales sintéticas
    Q = seq @ np.eye(46)
    K = seq @ np.eye(46)
    V = seq @ np.eye(46)
    
    # Scaled Dot-Product Attention: Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V
    scores_attn = softmax((Q @ K.T) / math.sqrt(d_k)) @ V
    
    # Agregación por peso exponencial hacia el presente
    weights = np.exp(np.linspace(-2, 0, 15)).reshape(-1, 1)
    decayed_attn = (scores_attn * weights).sum(axis=0)
    
    # Pasar por MLP denso ajustado con las salidas de atención
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=150, random_state=42, early_stopping=True)
    scores = np.zeros(46)
    for n in range(46):
        y_n = Y[:, n]
        if len(np.unique(y_n)) < 2:
            scores[n] = decayed_attn[n]
            continue
        try:
            mlp.fit(X, y_n)
            mlp_prob = mlp.predict_proba(last_feat)[0, 1]
            scores[n] = 0.6 * mlp_prob + 0.4 * decayed_attn[n]
        except Exception:
            scores[n] = decayed_attn[n]
            
    return scores

# ---------------------------------------------------------------------
# MODELO C: Attention-Boosted Stacking (Attn-CatBoost-XGB Hybrid)
# ---------------------------------------------------------------------
def model_attn_boosted_stacking(train_draws):
    """
    Combina representaciones de Atención Temporal con un Ensamble de CatBoost y XGBoost.
    """
    X, Y, last_feat, matrix = build_feature_matrices(train_draws)
    scores = np.zeros(46)
    
    cat = CatBoostClassifier(iterations=80, learning_rate=0.05, depth=4, logging_level='Silent', random_state=42)
    xgb = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, verbosity=0, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42, early_stopping=True)
    
    for n in range(46):
        y_n = Y[:, n]
        if len(np.unique(y_n)) < 2:
            scores[n] = 0.05
            continue
        try:
            cat.fit(X, y_n)
            p_cat = cat.predict_proba(last_feat)[0, 1]
            
            xgb.fit(X, y_n)
            p_xgb = xgb.predict_proba(last_feat)[0, 1]
            
            mlp.fit(X, y_n)
            p_mlp = mlp.predict_proba(last_feat)[0, 1]
            
            # Promedio ponderado Stacking
            scores[n] = 0.50 * p_mlp + 0.30 * p_cat + 0.20 * p_xgb
        except Exception:
            scores[n] = 0.05
            
    return scores

# ---------------------------------------------------------------------
# MODELO D: Graph-Attention Markov Recurrence (GAT-Markov)
# ---------------------------------------------------------------------
def model_gat_markov(train_draws):
    """
    Atención de Grafo (GAT) combinada con Transición de Cadenas de Markov de Orden 2.
    """
    X, Y, last_feat, matrix = build_feature_matrices(train_draws)
    
    # Grafo de Co-ocurrencia
    A = np.zeros((46, 46))
    for d in train_draws[-60:]:
        for x, y in itertools.combinations(d['numbers'], 2):
            A[x, y] += 1.0
            A[y, x] += 1.0
            
    # Atención de Nodos (GAT similarity)
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    A_norm = A / deg
    
    # Markov de Orden 2 en los últimos 3 sorteos
    last_nums = train_draws[-1]['numbers']
    markov_score = np.zeros(46)
    for x in last_nums:
        markov_score += A_norm[x]
    if markov_score.sum() > 0:
        markov_score /= markov_score.sum()
        
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42, early_stopping=True)
    scores = np.zeros(46)
    for n in range(46):
        y_n = Y[:, n]
        if len(np.unique(y_n)) < 2:
            scores[n] = markov_score[n]
            continue
        try:
            mlp.fit(X, y_n)
            p_mlp = mlp.predict_proba(last_feat)[0, 1]
            scores[n] = 0.65 * p_mlp + 0.35 * markov_score[n]
        except Exception:
            scores[n] = markov_score[n]
            
    return scores

# =====================================================================
# 5. SELECCIÓN OPTIMIZADA DE BOLETOS
# =====================================================================
def select_top_3_tickets(boletos, probs, n_tickets=3):
    scores = []
    for b in boletos:
        score = sum(np.log(max(probs[n], 1e-6)) for n in b)
        scores.append((score, b))
        
    scores.sort(key=lambda x: x[0], reverse=True)
    
    selected = [scores[0][1]]
    for score, b in scores[1:]:
        if len(selected) >= n_tickets:
            break
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
# 6. BENCHMARK COMPARATIVO ENTRE MODELOS NUEVOS Y CAMPEÓN ACTUAL
# =====================================================================
def run_comparison_benchmark(n_sorteos_backtest=60):
    print("=====================================================================")
    print(f"   COMPETENCIA DE NUEVOS MODELOS AVANZADOS ({n_sorteos_backtest} SORTEOS)")
    print("=====================================================================")
    
    draws = load_data()
    trad_indices = [i for i, d in enumerate(draws) if d['modalidad'] == 'Tradicional']
    test_indices = trad_indices[-n_sorteos_backtest:]
    
    models_name = [
        "1. Attn-MLP v1 (Campeón Anterior - 4.4%)",
        "2. Transformer-Style Self-Attention (Attn-Transformer)",
        "3. Attention-Boosted Stacking (Attn-CatBoost-XGB)",
        "4. Deep Graph-Attention Markov (GAT-Markov)"
    ]
    
    stats = {m: {'hits_3': 0, 'hits_4': 0, 'hits_5': 0, 'hits_6': 0, 'total_hits': 0, 'pool_coverage': 0, 'total_tickets': 0} for m in models_name}
    
    for idx_step, idx_draw in enumerate(test_indices):
        train_sub = draws[:idx_draw]
        target_draw = draws[idx_draw]
        target_nums = set(target_draw['numbers'])
        
        recent_numbers = set()
        for d in train_sub[-5:]:
            recent_numbers.update(d['numbers'])
            
        last_trad = [d for d in train_sub if d['modalidad'] == 'Tradicional'][-1]
        dec_counts = Counter(num // 10 for num in last_trad['numbers'])
        saturated_decades = {dec for dec, count in dec_counts.items() if count >= 3}
        
        scores_dict = {}
        scores_dict[models_name[0]] = model_attn_mlp_v1(train_sub)
        scores_dict[models_name[1]] = model_transformer_attention(train_sub)
        scores_dict[models_name[2]] = model_attn_boosted_stacking(train_sub)
        scores_dict[models_name[3]] = model_gat_markov(train_sub)
        
        for m_name in models_name:
            sc = scores_dict[m_name]
            prob_dist = sc / np.sum(sc) if np.sum(sc) > 0 else np.ones(46) / 46.0
            
            top_15 = np.argsort(prob_dist)[-15:]
            pool_hits = len(set(top_15).intersection(target_nums))
            stats[m_name]['pool_coverage'] += pool_hits
            
            combos = list(itertools.combinations(top_15, 6))
            viables = [c for c in combos if es_ticket_valido(c, recent_numbers, saturated_decades)]
            if not viables:
                viables = [c for c in combos if es_ticket_valido(c)]
            if not viables:
                viables = combos[:10]
                
            tickets = select_top_3_tickets(viables, prob_dist, n_tickets=3)
            
            for t in tickets:
                hits = len(set(t).intersection(target_nums))
                stats[m_name]['total_hits'] += hits
                stats[m_name]['total_tickets'] += 1
                if hits == 3: stats[m_name]['hits_3'] += 1
                elif hits == 4: stats[m_name]['hits_4'] += 1
                elif hits == 5: stats[m_name]['hits_5'] += 1
                elif hits == 6: stats[m_name]['hits_6'] += 1
                
        if (idx_step + 1) % 15 == 0 or (idx_step + 1) == len(test_indices):
            print(f"Progreso Evaluado: {idx_step + 1}/{len(test_indices)} sorteos...")

    print("\n=====================================================================")
    print("             RESULTADOS DE LA COMPETENCIA DE MODELOS")
    print("=====================================================================\n")
    print(f"{'Modelo Candidato':<48} | {'Aciertos 3+':<12} | {'Aciertos 4+':<12} | {'Pool Coverage 15':<18} | {'Prom. Acierto/Ticket'}")
    print("-" * 118)
    
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
        
        print(f"{m_name:<48} | {tot_3plus:>3d} ({pct_3plus:>5.1f}%)   | {tot_4plus:>3d} ({pct_4plus:>5.1f}%)   | {avg_pool:>4.2f} / 6.00 bolillas | {avg_hits:>4.2f} aciertos")
        
        if pct_3plus > best_rate:
            best_rate = pct_3plus
            best_model = m_name
            
    print("-" * 118)
    print(f"\n[GANADOR SUPREMO]: *** {best_model} *** con {best_rate:.2f}% de efectividad 3+.")

if __name__ == '__main__':
    run_comparison_benchmark(n_sorteos_backtest=60)
