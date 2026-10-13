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

# Import functions from main generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generador_cobertura_ml import (
    load_data, extract_features_for_step, update_transition_counts,
    compute_cooccurrence_matrix, es_ticket_valido, compute_attention_mlp_scores
)

def run_prediction_until_sorteo(cutoff_sorteo):
    """
    Trains ML Ensemble models strictly up to cutoff_sorteo (inclusive)
    and predicts probabilities for the next draw.
    """
    all_draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    train_draws = [d for d in all_draws if d['sorteo'] <= cutoff_sorteo]
    
    print(f"\n[MODELO ML] Entrenando con {len(train_draws)} registros históricos (hasta Sorteo #{cutoff_sorteo})...")
    
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    transition2_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    transition_mod_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for i in range(len(train_draws)):
        update_transition_counts(train_draws, i, transition_counts, transition2_counts, transition_mod_counts)
        
    # Build ML training sets
    X_train = []
    y_train = []
    
    min_history = 40
    for t in range(min_history, len(train_draws)):
        target_draw = train_draws[t]
        target_set = set(target_draw['numbers'])
        target_mod = target_draw['modalidad']
        
        feats = extract_features_for_step(train_draws, t, transition_counts, transition2_counts, transition_mod_counts, target_mod)
        for n in range(46):
            label = 1 if n in target_set else 0
            X_train.append(feats[n])
            y_train.append(label)
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Classifiers
    clf_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, scale_pos_weight=6.0, use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1)
    clf_lgb = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15, scale_pos_weight=6.0, verbose=-1, random_state=42, n_jobs=-1)
    clf_cat = CatBoostClassifier(iterations=150, learning_rate=0.05, depth=4, auto_class_weights='Balanced', logging_level='Silent', random_state=42, thread_count=-1)
    clf_rf  = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    
    clf_xgb.fit(X_train_scaled, y_train)
    clf_lgb.fit(X_train_scaled, y_train)
    clf_cat.fit(X_train_scaled, y_train)
    clf_rf.fit(X_train_scaled, y_train)
    
    # Predict for step T = len(train_draws)
    t_target = len(train_draws)
    target_mod = 'Tradicional'
    feats_target = extract_features_for_step(train_draws, t_target, transition_counts, transition2_counts, transition_mod_counts, target_mod)
    X_target = np.array([feats_target[n] for n in range(46)])
    X_target_scaled = scaler.transform(X_target)
    
    p_xgb = clf_xgb.predict_proba(X_target_scaled)[:, 1]
    p_lgb = clf_lgb.predict_proba(X_target_scaled)[:, 1]
    p_cat = clf_cat.predict_proba(X_target_scaled)[:, 1]
    p_rf  = clf_rf.predict_proba(X_target_scaled)[:, 1]
    
    attn_scores = compute_attention_mlp_scores(train_draws)
    
    probs_ensemble = (0.25 * p_xgb + 0.25 * p_lgb + 0.20 * p_cat + 0.15 * p_rf + 0.15 * attn_scores)
    
    # Borda ranking with recency
    counts = Counter()
    for d in train_draws[-60:]:
        for n in d['numbers']:
            counts[n] += 1
            
    recency_scores = np.zeros(46)
    for idx_d, d in enumerate(train_draws[-30:]):
        w = np.exp((idx_d - 30) / 10.0)
        for n in d['numbers']:
            recency_scores[n] += w
            
    final_scores = probs_ensemble * 0.5 + (recency_scores / (recency_scores.max() + 1e-5)) * 0.3 + (np.array([counts[n] for n in range(46)]) / 60.0) * 0.2
    
    borda_ranking = list(np.argsort(final_scores)[::-1])
    top_14 = sorted(borda_ranking[:14])
    top_10 = sorted(borda_ranking[:10])
    
    # Cooccurrence matrix
    cooccur_mat = compute_cooccurrence_matrix(train_draws)
    
    # Tickets generation
    todas_combos = list(itertools.combinations(top_14, 6))
    boletos_viables = [c for c in todas_combos if es_ticket_valido(c)]
    if not boletos_viables:
        boletos_viables = todas_combos[:50]
        
    core_pool = set(borda_ranking[:8])
    surge_pool = set(top_14) - core_pool
    
    viables_core = [c for c in boletos_viables if sum(1 for n in c if n in core_pool) >= 4]
    if not viables_core: viables_core = boletos_viables
    ticket_A = max(viables_core, key=lambda b: sum(final_scores[n] for n in b) + 200.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(b, 2)))
    
    viables_B = [c for c in boletos_viables if set(c) != set(ticket_A)]
    if not viables_B: viables_B = boletos_viables
    ticket_B = max(viables_B, key=lambda b: sum(cooccur_mat[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(final_scores[n] for n in b))
    
    scores_C = final_scores.copy()
    for n in set(ticket_A).union(set(ticket_B)):
        scores_C[n] *= 0.5
    viables_C = [c for c in boletos_viables if set(c) != set(ticket_A) and set(c) != set(ticket_B)]
    if not viables_C: viables_C = boletos_viables
    ticket_C = max(viables_C, key=lambda b: sum(scores_C[n] for n in b) + 150.0 * sum(1 for n in b if n in surge_pool))
    
    return {
        'top_14': top_14,
        'top_10': top_10,
        'tickets': [
            {'nombre': 'SUPER TICKET A (CORE SPIKE)', 'numeros': sorted(list(ticket_A))},
            {'nombre': 'SUPER TICKET B (AFFINITY CLIQUE)', 'numeros': sorted(list(ticket_B))},
            {'nombre': 'SUPER TICKET C (HYPERGRAPH COVER)', 'numeros': sorted(list(ticket_C))}
        ]
    }

def main():
    print("="*70)
    print(" EVALUACIÓN DE PREDICCIÓN PARA EL SORTEO DEL MIÉRCOLES (07/10/2026 - SORTEO #3415)")
    print("="*70)
    
    # 1. Obtenemos datos reales del Sorteo #3415
    all_draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    draws_3415 = [d for d in all_draws if d['sorteo'] == 3415]
    
    sorteo_real = {'sorteo': 3415, 'fecha': '2026-10-07 (Miércoles)'}
    for d in draws_3415:
        sorteo_real[d['modalidad']] = d['numbers']
        
    print(f"\n[1] RESULTADOS REALES DEL SORTEO #{sorteo_real['sorteo']} ({sorteo_real['fecha']}):")
    for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
        print(f"  - {mod:12s}: {sorteo_real.get(mod, [])}")
        
    pozo_extra_set = set(sorteo_real['Tradicional'] + sorteo_real['Segunda'] + sorteo_real['Revancha'])
    print(f"  - Pozo Extra (17 nros únicos): {sorted(list(pozo_extra_set))}")
    
    # 2. EVALUACIÓN 1: PREDICCIÓN GUARDADA PREVIA EN 'proxima_prediccion.json'
    print("\n" + "="*70)
    print("[2] EVALUACIÓN DE LA PREDICCIÓN ALMACENADA (proxima_prediccion.json / Oct 2)")
    print("="*70)
    with open('proxima_prediccion.json', 'r', encoding='utf-8') as f:
        pred_saved = json.load(f)
        
    pool_saved = pred_saved['top_numeros']
    tickets_saved = pred_saved['tickets']
    
    print(f"Top Pool Registrado (14 números): {pool_saved}")
    for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
        nums = sorteo_real[mod]
        hits_p = [n for n in nums if n in pool_saved]
        print(f"\nModalidad {mod}: {nums}")
        print(f"  -> Aciertos en Pool ({len(hits_p)}/6): {hits_p}")
        for t in tickets_saved:
            matched = set(t['numeros']).intersection(set(nums))
            print(f"     - {t['nombre']}: {len(matched)} aciertos -> {sorted(list(matched))}")
            
    hits_extra_saved = [n for n in sorted(list(pozo_extra_set)) if n in pool_saved]
    print(f"\nPozo Extra (Aciertos en Pool {len(hits_extra_saved)}/17): {hits_extra_saved}")
    for t in tickets_saved:
        matched = set(t['numeros']).intersection(pozo_extra_set)
        print(f"     - {t['nombre']}: {len(matched)} aciertos en Pozo Extra -> {sorted(list(matched))}")

    # 3. EVALUACIÓN 2: PREDICCIÓN FRESCA ENTRENADA HASTA SORTEO #3414 (DOMINGO 04/10/2026)
    print("\n" + "="*70)
    print("[3] EVALUACIÓN DE MODELO ML RE-ENTRENADO CON HISTORIAL HASTA SORTEO #3414")
    print("="*70)
    
    pred_fresh = run_prediction_until_sorteo(3414)
    pool_fresh_14 = pred_fresh['top_14']
    tickets_fresh = pred_fresh['tickets']
    
    print(f"\nTop Pool 14 Calculado: {pool_fresh_14}")
    print(f"Top Pool 10 Calculado: {pred_fresh['top_10']}")
    
    for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
        nums = sorteo_real[mod]
        hits_p14 = [n for n in nums if n in pool_fresh_14]
        hits_p10 = [n for n in nums if n in pred_fresh['top_10']]
        print(f"\nModalidad {mod}: {nums}")
        print(f"  -> Aciertos en Pool 14 ({len(hits_p14)}/6): {hits_p14}")
        print(f"  -> Aciertos en Pool 10 ({len(hits_p10)}/6): {hits_p10}")
        for t in tickets_fresh:
            matched = set(t['numeros']).intersection(set(nums))
            print(f"     - {t['nombre']}: {len(matched)} aciertos -> {sorted(list(matched))}")
            
    hits_extra_fresh14 = [n for n in sorted(list(pozo_extra_set)) if n in pool_fresh_14]
    print(f"\nPozo Extra (Aciertos en Pool 14: {len(hits_extra_fresh14)} aciertos): {hits_extra_fresh14}")
    for t in tickets_fresh:
        matched = set(t['numeros']).intersection(pozo_extra_set)
        print(f"     - {t['nombre']}: {len(matched)} aciertos en Pozo Extra -> {sorted(list(matched))}")

if __name__ == '__main__':
    main()
