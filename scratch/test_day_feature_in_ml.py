import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from generador_cobertura_ml import load_data, extract_features_for_step

draws = load_data()
trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']

print("Evaluando impacto de características de Día de la Semana en los modelos ML...")

# Let's run a rolling out-of-fold evaluation on the last 200 Tradicional draws
test_draws = trad_draws[-200:]

# Baseline AUC vs Enhanced AUC (with is_sunday and day_freq)
auc_base_list = []
auc_enh_list = []

for idx in range(50, len(test_draws), 10):
    eval_d = test_draws[idx]
    train_pool = [d for d in draws if d['sorteo'] < eval_d['sorteo']]
    train_trad = [d for d in train_pool if d['modalidad'] == 'Tradicional']
    
    # We can check ROC-AUC of predicting the 6 winning balls out of 46
    # Target: 1 if ball in eval_d['numbers'], 0 otherwise
    y_true = np.array([1 if n in eval_d['numbers'] else 0 for n in range(46)])
    
    # Calculate features for Sunday vs Wednesday
    d_date = datetime.strptime(eval_d['fecha'], '%Y-%m-%d')
    is_sunday = 1.0 if d_date.weekday() == 6 else 0.0
    
    # Day-specific frequencies
    same_day_hist = [d for d in train_trad if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == d_date.weekday()]
    c_day = Counter(n for d in same_day_hist[-30:] for n in d['numbers'])
    c_all = Counter(n for d in train_trad[-30:] for n in d['numbers'])
    
    p_day = np.array([c_day[n] for n in range(46)], dtype=float)
    p_all = np.array([c_all[n] for n in range(46)], dtype=float)
    p_day /= (p_day.sum() + 1e-5)
    p_all /= (p_all.sum() + 1e-5)
    
    # AUC with day vs without day
    # Let's measure ranking quality
    auc_base = roc_auc_score(y_true, p_all)
    auc_enh  = roc_auc_score(y_true, 0.5 * p_all + 0.5 * p_day)
    
    auc_base_list.append(auc_base)
    auc_enh_list.append(auc_enh)

print(f"ROC-AUC Promedio Base (Solo Historia Global): {np.mean(auc_base_list):.4f}")
print(f"ROC-AUC Promedio con Enfoque de Día (Miér/Dom): {np.mean(auc_enh_list):.4f}")
print(f"Diferencia: {np.mean(auc_enh_list) - np.mean(auc_base_list):+.4f}")

