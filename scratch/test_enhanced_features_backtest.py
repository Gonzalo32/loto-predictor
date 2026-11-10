"""
Test: Feature Engineering Avanzado en el Super-Ensemble de ML
Incorporación de:
1. is_wednesday
2. weekday_freq_20
3. in_prev_ss (Arrastre de Siempre Sale previo)
4. prev_date_count (Resonancia multi-modal fecha previa)
5. mod_delay_ratio (Atraso relativo en modalidad objetivo)
Evaluado en 500 sorteos con walk-forward estricto.
"""
import os, sys, itertools, time
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_antagonist_pairs,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    select_balanced_pool_9
)
from cobertura_combinatoria import select_optimal_wheel_triplet
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores

def select_pool_9_elite(borda_scores, saturated_decades=None, max_per_decade=3):
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
            if len(pool) == 9: break
    if len(pool) < 9:
        for n in ranked:
            if n not in pool:
                pool.append(n)
                if len(pool) == 9: break
    return sorted(pool)

# Función de extracción de características base vs avanzada
def extract_enhanced_features(draws, t, target_mod, target_weekday):
    history = draws[:t]
    n_history = len(history)
    
    # 1. Base counts & delays
    last_seen = {n: -1 for n in range(46)}
    delays_hist = defaultdict(list)
    for idx, d in enumerate(history):
        for n in d['numbers']:
            if last_seen[n] != -1:
                delays_hist[n].append(idx - last_seen[n])
            last_seen[n] = idx
            
    mean_delays = {n: np.mean(delays_hist[n]) if delays_hist[n] else 7.6 for n in range(46)}
    
    # Modality specific
    mod_history = [d for d in history if d['modalidad'] == target_mod]
    n_mod = len(mod_history)
    last_seen_mod = {n: -1 for n in range(46)}
    mod_delays = defaultdict(list)
    for idx, d in enumerate(mod_history):
        for n in d['numbers']:
            if last_seen_mod[n] != -1:
                mod_delays[n].append(idx - last_seen_mod[n])
            last_seen_mod[n] = idx
            
    mean_mod_delays = {n: np.mean(mod_delays[n]) if mod_delays[n] else 7.6 for n in range(46)}
    
    # Previa fecha
    last_sorteo = max(d['sorteo'] for d in history) if history else 0
    last_date_draws = [d for d in history if d['sorteo'] == last_sorteo]
    res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
    prev_ss = set(n for d in last_date_draws if d['modalidad'] == 'SiempreSale' for n in d['numbers'])
    
    # Weekday frequency
    same_day = [d for d in history if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    day_counts_20 = Counter(n for d in same_day[-20:] for n in d['numbers'])
    
    feats = {}
    for n in range(46):
        delay = n_history - 1 - last_seen[n] if last_seen[n] != -1 else n_history
        delay_ratio = delay / (mean_delays[n] + 1e-5)
        
        mod_delay = n_mod - 1 - last_seen_mod[n] if last_seen_mod[n] != -1 else n_mod
        mod_delay_ratio = mod_delay / (mean_mod_delays[n] + 1e-5)
        
        f5 = sum(1 for d in history[-5:] if n in d['numbers']) / 5.0 if n_history >= 5 else 0.0
        f20 = sum(1 for d in history[-20:] if n in d['numbers']) / 20.0 if n_history >= 20 else 0.0
        
        in_ss = 1.0 if n in prev_ss else 0.0
        p_cnt = float(res_counts[n])
        is_wed = 1.0 if target_weekday == 2 else 0.0
        day_f20 = day_counts_20[n] / 20.0
        
        feats[n] = [
            delay, delay_ratio,
            mod_delay, mod_delay_ratio,
            f5, f20,
            in_ss, p_cnt,
            is_wed, day_f20
        ]
    return feats

def run_fe_benchmark(n_eval=500):
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]
    
    print("="*115)
    print(f" BENCHMARK: ENSEMBLE BASE vs ENSEMBLE CON FEATURE ENGINEERING AVANZADO EN {n_eval} SORTEOS")
    print("="*115)
    
    metrics = {
        'A. Base Actual (Borda Clásico + Guardián)': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []
        },
        'B. Con Feature Engineering ML Integrado': {
            'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []
        }
    }
    
    start_time = time.time()
    
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
        
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1
            
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency_scores[n] += weight
            
        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)
        attn_scores = compute_fast_attention_mlp_scores(train_draws)
        
        trad_series = [d for d in train_draws if d['modalidad'] == 'Tradicional']
        last_5_trad = trad_series[-5:]
        n_rep = sum(len(set(last_5_trad[i]['numbers']) & set(last_5_trad[i-1]['numbers'])) for i in range(1, len(last_5_trad)))
        avg_rep = n_rep / 4.0
        if avg_rep >= 1.0: w_rec, w_ddm = 2.8, 1.2
        elif avg_rep <= 0.5: w_rec, w_ddm = 1.2, 2.8
        else: w_rec, w_ddm = 2.0, 1.8
        
        last_sorteo_num = max(d['sorteo'] for d in train_draws)
        last_date_draws = [d for d in train_draws if d['sorteo'] == last_sorteo_num]
        res_counts = Counter(n for d in last_date_draws for n in d['numbers'])
        prev_ss_nums = set()
        for d in last_date_draws:
            if d['modalidad'] == 'SiempreSale':
                prev_ss_nums.update(d['numbers'])
                
        saturated_decades = set()
        last_trad = trad_series[-1]
        for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
            if cnt >= 3: saturated_decades.add(dec)
            
        antag = compute_antagonist_pairs(train_draws)
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
        counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
        
        # 1. Borda Base A
        borda_a = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda_a[n] += w * (45 - rank)
            
        for n in range(46):
            if res_counts[n] >= 2: borda_a[n] += 18.0 * (res_counts[n] - 1)
            elif res_counts[n] == 1: borda_a[n] += 4.0
            if n in prev_ss_nums: borda_a[n] += 4.0
            
        # 2. Borda Avanzada B con Enhanced Features
        # Entrenar un LightGBM rápido en los últimos 60 sorteos con las nuevas features
        fe_t = extract_enhanced_features(train_draws, len(train_draws), 'Tradicional', target_weekday)
        X_test_b = np.array([fe_t[n] for n in range(46)])
        
        # Train dataset sobre últimos 60 sorteos
        X_tr, y_tr = [], []
        sub_len = len(train_draws)
        start_idx = max(20, sub_len - 60)
        for idx in range(start_idx, sub_len, 4):
            d_i = train_draws[idx]
            w_i = datetime.strptime(d_i['fecha'], '%Y-%m-%d').weekday()
            fe_i = extract_enhanced_features(train_draws, idx, d_i['modalidad'], w_i)
            target_nums = set(d_i['numbers'])
            for n in range(46):
                X_tr.append(fe_i[n])
                y_tr.append(1 if n in target_nums else 0)
                
        lgb_fast = LGBMClassifier(n_estimators=30, learning_rate=0.08, max_depth=3, verbose=-1, random_state=42, n_jobs=1)
        lgb_fast.fit(np.array(X_tr), np.array(y_tr))
        p_lgb_enhanced = lgb_fast.predict_proba(X_test_b)[:, 1]
        
        borda_b = borda_a.copy()
        rk_lgb = sorted(range(46), key=lambda x: p_lgb_enhanced[x], reverse=True)
        for rank, n in enumerate(rk_lgb):
            borda_b[n] += 1.6 * (45 - rank)
            
        def evaluate_borda(borda_dict):
            b_day = defaultdict(float)
            b_hib = defaultdict(float)
            for n in range(46):
                b_day[n] = borda_dict[n] + 16.0 * counts_day[n]
                b_hib[n] = borda_dict[n] + 8.0 * counts_day[n]
                
            p_base = select_pool_9_elite(borda_dict, saturated_decades)
            p_day = select_pool_9_elite(b_day, saturated_decades)
            p_hib = select_pool_9_elite(b_hib, saturated_decades)
            
            def get_v(p):
                c_list = list(itertools.combinations(p, 6))
                v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=3)]
                if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, min_distinct_decades=3)]
                return v or c_list[:10]
                
            v_b = get_v(p_base)
            v_d = get_v(p_day)
            v_h = get_v(p_hib)
            
            t_b = select_optimal_wheel_triplet(v_b, p_base, borda_dict, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
            
            b1 = t_b[0]
            cands_b2 = [c for c in v_d if 2 <= len(set(c) & set(b1)) <= 4]
            if not cands_b2: cands_b2 = [c for c in v_d if c != b1]
            b2 = max(cands_b2, key=lambda c: sum(b_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
            
            q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
            cands_b3 = [
                c for c in v_h
                if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2
            ]
            if not cands_b3: cands_b3 = [c for c in v_h if c != b1 and c != b2 and len(set(c) & set(b1)) <= 4 and len(set(c) & set(b2)) <= 4]
            if not cands_b3: cands_b3 = [c for c in v_h if c != b1 and c != b2]
            
            b3 = max(
                cands_b3,
                key=lambda c: sum(b_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12)
            )
            return [b1, b2, b3]

        t_a = evaluate_borda(borda_a)
        t_b = evaluate_borda(borda_b)
        
        for name, t_list in [('A. Base Actual (Borda Clásico + Guardián)', t_a),
                             ('B. Con Feature Engineering ML Integrado', t_b)]:
            hits_trad = [len(set(t) & target_trad) for t in t_list]
            max_h_trad = max(hits_trad)
            max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
            max_pe = max(len(set(t) & pozo_extra) for t in t_list)
            
            metrics[name]['max_trad'].append(max_h_trad)
            if max_h_trad >= 3: metrics[name]['trad_3+'] += 1
            if max_h_trad >= 4: metrics[name]['trad_4+'] += 1
            if max_any >= 4: metrics[name]['any_4+'] += 1
            if max_pe >= 5: metrics[name]['pe_5+'] += 1
            
        if (step_i + 1) % 100 == 0:
            print(f" -> Progreso: {step_i + 1} de {n_eval} sorteos completados...")
            
    elapsed = time.time() - start_time
    print(f"\n[OK] Benchmark completado en {elapsed:.1f} segundos.\n")
    
    print("="*125)
    print(f"{'Estrategia Evaluada (500 Sorteos)':<65} | {'Prom. Trad.':<10} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*125)
    for name, s in metrics.items():
        avg_t = np.mean(s['max_trad'])
        p3 = (s['trad_3+'] / n_eval) * 100
        n4_t = s['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = s['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = s['pe_5+']
        print(f"{name:<65} | {avg_t:<10.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*125)

if __name__ == '__main__':
    run_fe_benchmark(500)
