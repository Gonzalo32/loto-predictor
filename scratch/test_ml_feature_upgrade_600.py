import os, sys, itertools, time
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, compute_antagonist_pairs,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    select_optimal_wheel_triplet
)
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores
from scratch.compare_600_draws_varE import select_pool, es_ticket_valido_enhanced
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
n_eval = 300
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]

def compute_cross_modal_transitions(train_draws, window=60):
    sub = train_draws[-window:] if window else train_draws
    trans_matrix = np.zeros((46, 46))
    for i in range(1, len(sub)):
        for u in sub[i-1]['numbers']:
            for v in sub[i]['numbers']:
                trans_matrix[u][v] += 1.0
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return trans_matrix / row_sums

print(f"=== TEST COMPARATIVO 300 SORTEOS: INYECCION DE CARACTERISTICAS RESONANTES EN ML ===")

# Test 2 Configurations:
# C1: Baseline Actual (ML Estándar + Heurística Borda)
# C2: Enhanced ML (LGBM entrenado con 'last_date_count', 'in_siempre_sale', 'cross_pot' + Ensamble)

results = {
    '1. Baseline Actual': {'trad_3': 0, 'trad_4': 0, 'trad_5': 0, 'global_4plus': 0, 'pe_4plus': 0, 'pe_5plus': 0},
    '2. ML con Features de Resonancia Fecha': {'trad_3': 0, 'trad_4': 0, 'trad_5': 0, 'global_4plus': 0, 'pe_4plus': 0, 'pe_5plus': 0}
}

t0 = time.time()
for step_i, target_draw in enumerate(eval_draws):
    target_sorteo = target_draw['sorteo']
    target_fecha = target_draw['fecha']
    target_weekday = datetime.strptime(target_fecha, '%Y-%m-%d').weekday()
    group = [d for d in draws if d['sorteo'] == target_sorteo]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in group}
    target_trad = real_by_mod.get('Tradicional', set())
    target_seg = real_by_mod.get('Segunda', set())
    target_rev = real_by_mod.get('Revancha', set())
    target_ss = real_by_mod.get('SiempreSale', set())
    pozo_extra = target_trad | target_seg | target_rev
    train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
    
    counts_60 = Counter(n for d in train_draws[-60:] for n in d['numbers'])
    recency_scores = np.zeros(46)
    for t_idx, d in enumerate(train_draws[-30:]):
        weight = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency_scores[n] += weight
        
    pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws, window=40)
    ddm_scores = compute_delta_delay_momentum_scores(train_draws)
    attn_scores = compute_fast_attention_mlp_scores(train_draws)
    
    trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    last_5_trad = trad_series[-5:]
    n_rep = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
    avg_rep = n_rep / 4.0
    w_rec, w_ddm = (2.8, 1.2) if avg_rep >= 1.0 else ((1.2, 2.8) if avg_rep <= 0.5 else (2.0, 1.8))
    
    last_sorteo_num = max(d['sorteo'] for d in train_draws)
    last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
    res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
    prev_ss_nums = set(n for d in last_date_draws if d['modalidad'] == 'SiempreSale' for n in d['numbers'])
    
    saturated_decades = set()
    for dec, cnt in Counter(n // 10 for n in trad_series[-1]['numbers']).items():
        if cnt >= 3: saturated_decades.add(dec)
        
    antag = compute_antagonist_pairs(train_draws)
    recent_numbers = set(n for d in train_draws[-5:] for n in d['numbers'])
    same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
    
    cross_mat = compute_cross_modal_transitions(train_draws, window=60)
    cross_scores = np.zeros(46)
    for d_prev in last_date_draws:
        weight_m = 1.5 if d_prev['modalidad'] == 'SiempreSale' else 1.0
        for u in d_prev['numbers']:
            cross_scores += weight_m * cross_mat[u]
            
    # Config 1: Baseline
    borda_1 = defaultdict(float)
    models_1 = [
        (counts_60, 1.2),
        (recency_scores, w_rec * 1.3),
        (pcr_scores, 1.4),
        (ddm_scores, w_ddm * 0.8),
        ({n: attn_scores[n] for n in range(46)}, 2.0),
        (cross_scores, 1.5)
    ]
    for m, w in models_1:
        rk = sorted(range(46), key=lambda x: m[x], reverse=True)
        for rank, n in enumerate(rk):
            borda_1[n] += w * (45 - rank)
    for n in range(46):
        if res_counts[n] >= 2: borda_1[n] += 18.0 * (res_counts[n] - 1)
        elif res_counts[n] == 1: borda_1[n] += 4.0
        if n in prev_ss_nums: borda_1[n] += 4.0

    # Config 2: Enhanced ML (LightGBM on top of dynamic cross features)
    # Train a quick LightGBM on the last 40 date intervals
    # Features per number: [counts_60, recency, pcr, ddm, attn, cross, res_count, in_ss]
    X_hist, y_hist = [], []
    # Train on recent historical steps
    sorteos_train = sorted(list(set(d['sorteo'] for d in train_draws)))[-35:]
    for s_idx in range(5, len(sorteos_train)):
        s_target = sorteos_train[s_idx]
        s_prev = sorteos_train[s_idx-1]
        s_prev_draws = [d for d in train_draws if d['sorteo'] == s_prev]
        s_target_trad = [d for d in train_draws if d['sorteo'] == s_target and d['modalidad'] == 'Tradicional']
        if not s_target_trad: continue
        real_set = set(s_target_trad[0]['numbers'])
        
        c_prev_date = Counter(n for d in s_prev_draws for n in d['numbers'])
        ss_prev = set(n for d in s_prev_draws if d['modalidad'] == 'SiempreSale' for n in d['numbers'])
        sub_train = [d for d in train_draws if d['sorteo'] < s_target]
        c60 = Counter(n for d in sub_train[-60:] for n in d['numbers'])
        
        for n in range(46):
            X_hist.append([
                c60[n],
                c_prev_date[n],
                1.0 if n in ss_prev else 0.0,
                1.0 if n in sub_train[-1]['numbers'] else 0.0
            ])
            y_hist.append(1 if n in real_set else 0)
            
    clf_quick = LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.08, verbose=-1, random_state=42, n_jobs=1)
    clf_quick.fit(X_hist, y_hist)
    
    X_now = [[counts_60[n], res_counts[n], 1.0 if n in prev_ss_nums else 0.0, 1.0 if n in train_draws[-1]['numbers'] else 0.0] for n in range(46)]
    lgbm_scores = clf_quick.predict_proba(X_now)[:, 1]
    
    borda_2 = defaultdict(float)
    models_2 = [
        (counts_60, 1.0),
        (recency_scores, w_rec * 1.2),
        (pcr_scores, 1.3),
        (ddm_scores, w_ddm * 0.8),
        ({n: attn_scores[n] for n in range(46)}, 1.8),
        (cross_scores, 1.4),
        ({n: lgbm_scores[n] for n in range(46)}, 2.0)
    ]
    for m, w in models_2:
        rk = sorted(range(46), key=lambda x: m[x], reverse=True)
        for rank, n in enumerate(rk):
            borda_2[n] += w * (45 - rank)
    for n in range(46):
        if res_counts[n] >= 2: borda_2[n] += 12.0 * (res_counts[n] - 1)
        elif res_counts[n] == 1: borda_2[n] += 3.0
        if n in prev_ss_nums: borda_2[n] += 3.0

    eval_pairs = [
        ('1. Baseline Actual', borda_1),
        ('2. ML con Features de Resonancia Fecha', borda_2)
    ]

    for name, b_map in eval_pairs:
        b_day = defaultdict(float)
        b_hib = defaultdict(float)
        for n in range(46):
            b_day[n] = b_map[n] + 16.0 * counts_day[n]
            b_hib[n] = b_map[n] + 8.0 * counts_day[n]
            
        p_base = select_pool(b_map, saturated_decades, pool_size=9)
        p_day = select_pool(b_day, saturated_decades, pool_size=9)
        p_hib = select_pool(b_hib, saturated_decades, pool_size=10)
        
        def get_v(p):
            c_list = list(itertools.combinations(p, 6))
            v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, antag, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
            if len(v) < 3: v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, None, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
            return v or c_list[:10]
            
        vb, vd, vh = get_v(p_base), get_v(p_day), get_v(p_hib)
        b1 = select_optimal_wheel_triplet(vb, p_base, b_map, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)[0]
        cands_b2 = [c for c in vd if 2 <= len(set(c) & set(b1)) <= 4] or [c for c in vd if c != b1]
        b2 = max(cands_b2, key=lambda c: sum(b_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
        cands_b3 = [c for c in vh if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2] or [c for c in vh if c != b1 and c != b2 and len(set(c) & set(b1)) <= 4 and len(set(c) & set(b2)) <= 4] or [c for c in vh if c != b1 and c != b2]
        b3 = max(cands_b3, key=lambda c: sum(b_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12))
        
        tickets = [b1, b2, b3]
        h_trad = max(len(set(t) & target_trad) for t in tickets)
        h_seg  = max(len(set(t) & target_seg) for t in tickets) if target_seg else 0
        h_rev  = max(len(set(t) & target_rev) for t in tickets) if target_rev else 0
        h_ss   = max(len(set(t) & target_ss) for t in tickets) if target_ss else 0
        h_pe   = max(len(set(t) & pozo_extra) for t in tickets)
        
        res = results[name]
        if h_trad >= 3: res['trad_3'] += 1
        if h_trad >= 4: res['trad_4'] += 1
        if h_trad >= 5: res['trad_5'] += 1
        if max(h_trad, h_seg, h_rev, h_ss) >= 4: res['global_4plus'] += 1
        if h_pe >= 4: res['pe_4plus'] += 1
        if h_pe >= 5: res['pe_5plus'] += 1

    if (step_i + 1) % 50 == 0:
        print(f"[{step_i + 1}/{n_eval}] sorteos completados ({time.time()-t0:.1f}s)...")

print("\n" + "="*115)
print(f"{'CONFIGURACION (300 SORTEOS)':<45} | {'TRAD 3+':<8} | {'TRAD 4+':<8} | {'GLOB 4+':<8} | {'PE 4+':<8} | {'PE 5+':<8}")
print("="*115)
for name, res in results.items():
    print(f"{name:<45} | {res['trad_3']:>2} ({res['trad_3']/n_eval*100:>4.1f}%) | {res['trad_4']:>2} ({res['trad_4']/n_eval*100:>3.1f}%) | {res['global_4plus']:>2} ({res['global_4plus']/n_eval*100:>3.1f}%) | {res['pe_4plus']:>2} ({res['pe_4plus']/n_eval*100:>4.1f}%) | {res['pe_5plus']:>2} ({res['pe_5plus']/n_eval*100:>3.1f}%)")
print("="*115)
