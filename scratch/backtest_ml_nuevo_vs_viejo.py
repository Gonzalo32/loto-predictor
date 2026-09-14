import csv
import json
import numpy as np
import itertools
import os
import sys
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

def load_data(filepath='historico_quini_completo.csv'):
    if not os.path.exists(filepath):
        filepath = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
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
            except (ValueError, KeyError):
                continue
    modality_order = {'Tradicional': 1, 'Segunda': 2, 'Revancha': 3, 'SiempreSale': 4}
    draws.sort(key=lambda x: (x['sorteo'], modality_order.get(x['modalidad'], 9)))
    return draws

def es_ticket_valido(ticket):
    ticket = sorted(ticket)
    gaps_le_1 = sum(1 for i in range(1, len(ticket)) if ticket[i] - ticket[i-1] <= 1)
    if gaps_le_1 > 2:
        return False
    spread = ticket[5] - ticket[0]
    if spread < 15 or spread > 45:
        return False
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(c > 4 for c in decenas.values()):
        return False
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares == 0 or pares == 6:
        return False
    return True

def compute_attention_mlp_scores(train_draws):
    trad_draws = [d['numbers'] for d in train_draws if d['modalidad'] == 'Tradicional']
    if len(trad_draws) < 30:
        trad_draws = [d['numbers'] for d in train_draws]
    T = len(trad_draws)
    matrix = np.zeros((T, 46))
    for t, nums in enumerate(trad_draws):
        for n in nums:
            matrix[t, n] = 1.0
    X_list, y_list = [], []
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
    scores = np.zeros(46)
    mlp_attn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42, early_stopping=True)
    for n in range(46):
        y_n = Y[:, n]
        if len(np.unique(y_n)) < 2:
            scores[n] = 0.05
            continue
        try:
            mlp_attn.fit(X, y_n)
            scores[n] = mlp_attn.predict_proba(last_feat)[0, 1]
        except Exception:
            scores[n] = 0.05
    return scores

def update_transition_counts(draws, i, transition_counts, transition2_counts, transition_mod_counts):
    if i > 0:
        d_prev, d_curr = draws[i-1], draws[i]
        key = (d_prev['modalidad'], d_curr['modalidad'])
        for x in d_prev['numbers']:
            for y in d_curr['numbers']:
                transition_counts[key][x][y] += 1
    if i > 1:
        d_prev2, d_curr = draws[i-2], draws[i]
        key = (d_prev2['modalidad'], d_curr['modalidad'])
        for x in d_prev2['numbers']:
            for y in d_curr['numbers']:
                transition2_counts[key][x][y] += 1
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
    last_seen = {n: -1 for n in range(46)}
    delays_hist = defaultdict(list)
    for idx, d in enumerate(history):
        for n in d['numbers']:
            if last_seen[n] != -1:
                delays_hist[n].append(idx - last_seen[n])
            last_seen[n] = idx
    mean_delays = {n: (np.mean(delays_hist[n]) if delays_hist[n] else 7.6) for n in range(46)}
    current_delays = {n: (n_history - 1 - last_seen[n] if last_seen[n] != -1 else n_history) for n in range(46)}
    freq_5 = Counter()
    for d in history[-5:]:
        for n in d['numbers']:
            freq_5[n] += 1
    freq_20 = Counter()
    for d in history[-20:]:
        for n in d['numbers']:
            freq_20[n] += 1
    freq_50 = Counter()
    for d in history[-50:]:
        for n in d['numbers']:
            freq_50[n] += 1
    features = {}
    for n in range(46):
        f5 = freq_5[n] / 30.0
        f20 = freq_20[n] / 120.0
        f50 = freq_50[n] / 300.0
        cdelay = current_delays[n]
        mdelay = mean_delays[n]
        normalized_delay = cdelay / mdelay if mdelay > 0 else 1.0
        features[n] = [
            f5, f20, f50, cdelay, mdelay, normalized_delay,
            mod_numeric, 0.0, 0.0, 0.0
        ]
    return features

def compute_cooccurrence_matrix(draws):
    matrix = np.zeros((46, 46))
    total = len(draws) or 1
    for d in draws:
        nums = d['numbers']
        for u, v in itertools.combinations(nums, 2):
            matrix[u, v] += 1.0
            matrix[v, u] += 1.0
    return matrix / total

def select_old_tickets(boletos_viables, borda_scores):
    tickets_old = []
    curr_b = borda_scores.copy()
    for _ in range(3):
        best_t = max(boletos_viables, key=lambda b: sum(curr_b[n] for n in b))
        tickets_old.append(best_t)
        for n in best_t:
            curr_b[n] *= 0.0
    return tickets_old

def select_new_tickets(boletos_viables, borda_scores, cooccur_matrix, gamma=0.75, w_co=250.0):
    ticket_A = max(boletos_viables, key=lambda b: sum(borda_scores[n] for n in b) + w_co * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)))
    borda_B = borda_scores.copy()
    for n in ticket_A:
        borda_B[n] *= gamma
    ticket_B = max(boletos_viables, key=lambda b: sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda_B[n] for n in b))
    borda_C = borda_B.copy()
    for n in ticket_B:
        borda_C[n] *= gamma
    ticket_C = max(boletos_viables, key=lambda b: sum(borda_C[n] for n in b) + (w_co * 0.8) * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)))
    return [ticket_A, ticket_B, ticket_C]

def main():
    print("==================================================")
    print("BACKTEST COMPARATIVO DE APRENDIZAJE AUTOMÁTICO (ML)")
    print("Super-Ensemble Stacking + Attention Neural Network Engine")
    print("==================================================\n")
    
    draws = load_data()
    n_draws = len(draws)
    print(f"Cargados {n_draws} sub-sorteos históricos.")
    
    # Backtest over last 50 sub-sorteos (12.5 sorteos completos de las 4 modalidades)
    eval_steps = 40
    start_eval = n_draws - eval_steps
    
    hits_old_dist = defaultdict(int)
    hits_new_dist = defaultdict(int)
    
    max_old_list = []
    max_new_list = []
    pool_hits_list = []
    
    clf_xgb = XGBClassifier(n_estimators=60, learning_rate=0.05, max_depth=3, scale_pos_weight=6.0, use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1)
    clf_lgb = LGBMClassifier(n_estimators=60, learning_rate=0.05, max_depth=3, num_leaves=10, scale_pos_weight=6.0, verbose=-1, random_state=42, n_jobs=-1)
    clf_cat = CatBoostClassifier(iterations=80, learning_rate=0.05, depth=3, auto_class_weights='Balanced', logging_level='Silent', random_state=42, thread_count=-1)
    clf_rf  = RandomForestClassifier(n_estimators=60, max_depth=4, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42)
    clf_meta = LogisticRegression(random_state=42)
    
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    transition2_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    transition_mod_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for i in range(start_eval):
        update_transition_counts(draws, i, transition_counts, transition2_counts, transition_mod_counts)
        
    for step_idx, t in enumerate(range(start_eval, n_draws)):
        target_draw = set(draws[t]['numbers'])
        target_mod = draws[t]['modalidad']
        
        # Build dataset up to t-1
        X_train_list, y_train_list = [], []
        train_start = max(10, t - 150)
        for i in range(train_start, t):
            feats_i = extract_features_for_step(draws, i, transition_counts, transition2_counts, transition_mod_counts, draws[i]['modalidad'])
            real_i = set(draws[i]['numbers'])
            for n in range(46):
                X_train_list.append(feats_i[n])
                y_train_list.append(1 if n in real_i else 0)
                
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf_xgb.fit(X_train, y_train)
        clf_lgb.fit(X_train, y_train)
        clf_cat.fit(X_train, y_train)
        clf_rf.fit(X_train, y_train)
        clf_mlp.fit(X_train_scaled, y_train)
        
        preds_xgb_tr = clf_xgb.predict_proba(X_train)[:, 1]
        preds_lgb_tr = clf_lgb.predict_proba(X_train)[:, 1]
        preds_cat_tr = clf_cat.predict_proba(X_train)[:, 1]
        preds_rf_tr  = clf_rf.predict_proba(X_train)[:, 1]
        preds_mlp_tr = clf_mlp.predict_proba(X_train_scaled)[:, 1]
        
        X_meta_tr = np.column_stack([preds_xgb_tr, preds_lgb_tr, preds_cat_tr, preds_rf_tr, preds_mlp_tr])
        clf_meta.fit(X_meta_tr, y_train)
        
        feats_t = extract_features_for_step(draws, t, transition_counts, transition2_counts, transition_mod_counts, target_mod)
        X_test = np.array([feats_t[n] for n in range(46)])
        X_test_scaled = scaler.transform(X_test)
        
        p_xgb = clf_xgb.predict_proba(X_test)[:, 1]
        p_lgb = clf_lgb.predict_proba(X_test)[:, 1]
        p_cat = clf_cat.predict_proba(X_test)[:, 1]
        p_rf  = clf_rf.predict_proba(X_test)[:, 1]
        p_mlp = clf_mlp.predict_proba(X_test_scaled)[:, 1]
        
        p_attn = compute_attention_mlp_scores(draws[:t])
        X_meta_te = np.column_stack([p_xgb, p_lgb, p_cat, p_rf, p_mlp])
        probs_t = clf_meta.predict_proba(X_meta_te)[:, 1]
        
        borda = defaultdict(float)
        models_to_rank = [(p_attn, 3.0), (probs_t, 1.5), (p_mlp, 1.0), (p_cat, 1.0), (p_xgb, 1.0), (p_lgb, 1.0), (p_rf, 1.0)]
        for p_model, weight in models_to_rank:
            ranking_model = sorted(range(46), key=lambda x: p_model[x], reverse=True)
            for rank, n in enumerate(ranking_model):
                borda[n] += weight * (45 - rank)
                
        borda_ranking = [n for n, score in sorted(borda.items(), key=lambda x: x[1], reverse=True)]
        top_pool = borda_ranking[:15]
        
        pool_hits = len(set(top_pool).intersection(target_draw))
        pool_hits_list.append(pool_hits)
        
        todas_comb = list(itertools.combinations(sorted(top_pool), 6))
        boletos_viables = [combo for combo in todas_comb if es_ticket_valido(combo)]
        if not boletos_viables:
            boletos_viables = todas_comb[:100]
            
        cooccur_matrix = compute_cooccurrence_matrix(draws[:t])
        
        tickets_old = select_old_tickets(boletos_viables, borda)
        max_old = 0
        for b in tickets_old:
            h = len(set(b).intersection(target_draw))
            hits_old_dist[h] += 1
            max_old = max(max_old, h)
        max_old_list.append(max_old)
        
        tickets_new = select_new_tickets(boletos_viables, borda, cooccur_matrix, gamma=0.75, w_co=250.0)
        max_new = 0
        for b in tickets_new:
            h = len(set(b).intersection(target_draw))
            hits_new_dist[h] += 1
            max_new = max(max_new, h)
        max_new_list.append(max_new)
        
        update_transition_counts(draws, t, transition_counts, transition2_counts, transition_mod_counts)
        print(f"Paso [{step_idx+1:02d}/{eval_steps}] Sorteo {draws[t]['sorteo']} ({target_mod}): Pool Hits = {pool_hits}/6 | Max Ticket Viejo = {max_old} | Max Ticket Nuevo = {max_new}")

    print("\n==================================================")
    print("RESULTADOS FINALES DEL BACKTEST ML (WALK-FORWARD)")
    print("==================================================")
    tot_tickets = eval_steps * 3
    print("\n1. DISTRIBUCIÓN DE ACIERTOS POR BOLETO GENERADO:")
    print("Aciertos | Modelo Viejo (Disyunto) | Modelo Nuevo (Co-ocurrencia) | Variación Absoluta")
    print("---------+------------------------+-----------------------------+-------------------")
    for h in range(7):
        n_old = hits_old_dist[h]
        n_new = hits_new_dist[h]
        pct_old = (n_old / tot_tickets) * 100
        pct_new = (n_new / tot_tickets) * 100
        diff = pct_new - pct_old
        sign = "+" if diff > 0 else ""
        print(f" {h} Hits  | {n_old:4d} ({pct_old:6.2f}%)        | {n_new:4d} ({pct_new:6.2f}%)        | {sign}{diff:+.2f}%")
        
    print("---------+------------------------+-----------------------------+-------------------")
    old_3plus = sum(hits_old_dist[h] for h in range(3, 7))
    new_3plus = sum(hits_new_dist[h] for h in range(3, 7))
    print(f" 3+ Hits | {old_3plus:4d} ({(old_3plus/tot_tickets)*100:6.2f}%)        | {new_3plus:4d} ({(new_3plus/tot_tickets)*100:6.2f}%)        | {((new_3plus - old_3plus)/tot_tickets)*100:+.2f}%")

    print("\n2. CONCENTRACIÓN DE ACIERTOS MÁXIMOS EN UN SOLO TICKET (POR SORTEO):")
    old_draws_3plus = sum(1 for m in max_old_list if m >= 3)
    new_draws_3plus = sum(1 for m in max_new_list if m >= 3)
    old_draws_4plus = sum(1 for m in max_old_list if m >= 4)
    new_draws_4plus = sum(1 for m in max_new_list if m >= 4)
    
    print(f"- Sorteos con al menos 1 ticket de 3+ aciertos: Modelo Viejo = {old_draws_3plus}/{eval_steps} ({(old_draws_3plus/eval_steps)*100:.1f}%) | Modelo Nuevo = {new_draws_3plus}/{eval_steps} ({(new_draws_3plus/eval_steps)*100:.1f}%)")
    print(f"- Sorteos con al menos 1 ticket de 4+ aciertos: Modelo Viejo = {old_draws_4plus}/{eval_steps} ({(old_draws_4plus/eval_steps)*100:.1f}%) | Modelo Nuevo = {new_draws_4plus}/{eval_steps} ({(new_draws_4plus/eval_steps)*100:.1f}%)")
    
    avg_pool = np.mean(pool_hits_list)
    print(f"\n- Promedio de aciertos en el Pool de 15 números: {avg_pool:.2f} aciertos por sorteo")
    print("==================================================")

if __name__ == '__main__':
    main()
