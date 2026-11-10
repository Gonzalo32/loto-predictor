import os

file_path = os.path.join(os.getcwd(), 'generador_cobertura_ml.py')
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update fast multi-output attention MLP
target1 = """    scores = np.zeros(46)
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
            
    return scores"""

replacement1 = """    try:
        mlp_attn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=80, random_state=42, early_stopping=True)
        mlp_attn.fit(X, Y)
        scores = mlp_attn.predict_proba(last_feat)[0]
    except Exception:
        scores = np.mean(matrix[-20:], axis=0)
    return scores"""

content_norm = content.replace('\r\n', '\n')
if target1 in content_norm:
    content_norm = content_norm.replace(target1, replacement1, 1)
    print("Target 1 (Fast Attention MLP) replaced successfully")
else:
    print("Target 1 NOT found (might already be replaced)")

# 2. Update dynamic regime ensemble weights in main()
target2 = """    # Fusión Borda Count ponderada con prioridad para el motor de Atención Temporal, PCR, DDM y Cross-Modal
    borda = defaultdict(float)
    models_to_rank = [
        (p_attn, 3.0),
        (probs_t, 1.5),
        (pcr_scores, 1.5),
        (cross_scores, 1.2),
        (ddm_scores, 1.8),
        (p_mlp, 1.0),
        (p_cat, 1.0),
        (p_xgb, 1.0),
        (p_lgb, 1.0),
        (p_rf, 1.0)
    ]"""

replacement2 = """    # -------------------------------------------------------------
    # DETECCIÓN DE RÉGIMEN DINÁMICO (IDEA 2: MOMENTUM VS REVERSIÓN)
    # Analiza la tasa de repetición en los últimos 5 sorteos
    # -------------------------------------------------------------
    last_5 = draws[-5:]
    n_repeats = sum(len(set(last_5[i]['numbers']) & set(last_5[i-1]['numbers'])) for i in range(1, len(last_5)))
    avg_repeats = n_repeats / 4.0
    
    if avg_repeats >= 1.25:
        regime_desc = "Régimen Caliente / Alta Inercia (Prioridad Momentum/Recencia)"
        w_rec = 3.0
        w_ddm = 1.0
        w_attn = 3.2
    elif avg_repeats <= 0.75:
        regime_desc = "Régimen Frío / Retorno a la Media (Prioridad Atraso Relativo DDM)"
        w_rec = 1.0
        w_ddm = 3.0
        w_attn = 2.5
    else:
        regime_desc = "Régimen Balanceado (Equilibrio Momentum y Atraso)"
        w_rec = 2.0
        w_ddm = 1.8
        w_attn = 3.0
        
    print(f"\\n==================================================")
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
    ]"""

if target2 in content_norm:
    content_norm = content_norm.replace(target2, replacement2, 1)
    print("Target 2 (Dynamic Regime Weights) replaced successfully")
else:
    print("Target 2 NOT found")

target3 = '"juego": "Quini 6 (Super-Modelo Combinado - Adaptive Dynamic Pool + Cross-Modal Transitions)",'
replacement3 = '"juego": "Quini 6 (Super-Modelo Dinámico por Régimen + Rueda Balanceada 2-Fold BIBD)",'
if target3 in content_norm:
    content_norm = content_norm.replace(target3, replacement3, 1)
    print("Target 3 (JSON Description) replaced successfully")
else:
    print("Target 3 NOT found")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content_norm)

print("generador_cobertura_ml.py updated successfully with Idea 2!")

