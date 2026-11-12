import os, sys, itertools, time
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

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
n_eval = 300
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]

# Test 1: Full-Date Cross Transitions (All 24 numbers from prior date vs only last 6)
# Test 2: Double-counting removal in Borda (Only Stacking Meta vs Meta + Base models)
# Test 3: Cooccurrence window (40 vs 60 vs 80)

configs = {
    'Baseline (Producción Actual)': {
        'cross_mode': 'last6', 'borda_models': 'all', 'co_win': 40
    },
    'Var 1: Cross-Transitions Fecha Completa': {
        'cross_mode': 'full_date', 'borda_models': 'all', 'co_win': 40
    },
    'Var 2: Borda Pura (Meta + Attn + PCR + DDM)': {
        'cross_mode': 'last6', 'borda_models': 'pure_meta', 'co_win': 40
    },
    'Var 3: Coocurrencia Ventana 60': {
        'cross_mode': 'last6', 'borda_models': 'all', 'co_win': 60
    },
    'Var 4: Combinada (Full Cross + Co-Win 60)': {
        'cross_mode': 'full_date', 'borda_models': 'all', 'co_win': 60
    }
}

results = {name: {'trad_3': 0, 'trad_4': 0, 'global_4plus': 0, 'pe_4plus': 0, 'pe_5plus': 0} for name in configs}

def compute_cross_modal_transitions(train_draws):
    trans_matrix = np.zeros((46, 46))
    for i in range(1, len(train_draws)):
        for u in train_draws[i-1]['numbers']:
            for v in train_draws[i]['numbers']:
                trans_matrix[u][v] += 1.0
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return trans_matrix / row_sums

print(f"Iniciando evaluación de {len(configs)} variantes sobre {n_eval} sorteos...")
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
    
    cross_mat = compute_cross_modal_transitions(train_draws[-60:])
    
    for cfg_name, cfg in configs.items():
        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws, window=cfg['co_win'])
        
        # Cross scores
        cross_scores = np.zeros(46)
        if cfg['cross_mode'] == 'last6':
            for u in train_draws[-1]['numbers']:
                cross_scores += cross_mat[u]
        else: # full date
            for d_prev in last_date_draws:
                weight_m = 1.5 if d_prev['modalidad'] == 'SiempreSale' else 1.0
                for u in d_prev['numbers']:
                    cross_scores += weight_m * cross_mat[u]
                    
        # Borda ranking
        borda = defaultdict(float)
        if cfg['borda_models'] == 'all':
            models = [
                (counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4),
                (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5),
                (cross_scores, 1.2)
            ]
        else: # pure meta
            models = [
                (recency_scores, w_rec * 1.5), (pcr_scores, 1.5),
                (ddm_scores, w_ddm * 1.2), ({n: attn_scores[n] for n in range(46)}, 2.0),
                (cross_scores, 1.2)
            ]
            
        for m, w in models:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk):
                borda[n] += w * (45 - rank)
                
        for n in range(46):
            if res_counts[n] >= 2: borda[n] += 18.0 * (res_counts[n] - 1)
            elif res_counts[n] == 1: borda[n] += 4.0
            if n in prev_ss_nums: borda[n] += 4.0
            
        borda_day = defaultdict(float)
        borda_hib = defaultdict(float)
        for n in range(46):
            borda_day[n] = borda[n] + 16.0 * counts_day[n]
            borda_hib[n] = borda[n] + 8.0 * counts_day[n]
            
        p_base = select_pool(borda, saturated_decades, pool_size=9)
        p_day = select_pool(borda_day, saturated_decades, pool_size=9)
        p_hib = select_pool(borda_hib, saturated_decades, pool_size=10)
        
        def get_v(p):
            c_list = list(itertools.combinations(p, 6))
            v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, antag, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
            if len(v) < 3: v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, None, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
            return v or c_list[:10]
            
        vb, vd, vh = get_v(p_base), get_v(p_day), get_v(p_hib)
        b1 = select_optimal_wheel_triplet(vb, p_base, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)[0]
        cands_b2 = [c for c in vd if 2 <= len(set(c) & set(b1)) <= 4] or [c for c in vd if c != b1]
        b2 = max(cands_b2, key=lambda c: sum(borda_day[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
        cands_b3 = [c for c in vh if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2)) <= 4) and len(set(b1) & set(b2) & set(c)) <= 2] or [c for c in vh if c != b1 and c != b2 and len(set(c) & set(b1)) <= 4 and len(set(c) & set(b2)) <= 4] or [c for c in vh if c != b1 and c != b2]
        b3 = max(cands_b3, key=lambda c: sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12))
        
        tickets = [b1, b2, b3]
        h_trad = max(len(set(t) & target_trad) for t in tickets)
        h_seg  = max(len(set(t) & target_seg) for t in tickets) if target_seg else 0
        h_rev  = max(len(set(t) & target_rev) for t in tickets) if target_rev else 0
        h_ss   = max(len(set(t) & target_ss) for t in tickets) if target_ss else 0
        h_pe   = max(len(set(t) & pozo_extra) for t in tickets)
        
        res = results[cfg_name]
        if h_trad >= 3: res['trad_3'] += 1
        if h_trad >= 4: res['trad_4'] += 1
        if max(h_trad, h_seg, h_rev, h_ss) >= 4: res['global_4plus'] += 1
        if h_pe >= 4: res['pe_4plus'] += 1
        if h_pe >= 5: res['pe_5plus'] += 1

    if (step_i + 1) % 50 == 0:
        print(f"[{step_i + 1}/{n_eval}] completados en {time.time()-t0:.1f}s...")

print("\n" + "="*110)
print(f"{'CONFIGURACION':<42} | {'TRAD 3+':<8} | {'TRAD 4+':<8} | {'GLOB 4+':<8} | {'PE 4+':<8} | {'PE 5+':<8}")
print("="*110)
for name, res in results.items():
    print(f"{name:<42} | {res['trad_3']:>2} ({res['trad_3']/n_eval*100:>4.1f}%) | {res['trad_4']:>2} ({res['trad_4']/n_eval*100:>3.1f}%) | {res['global_4plus']:>2} ({res['global_4plus']/n_eval*100:>3.1f}%) | {res['pe_4plus']:>2} ({res['pe_4plus']/n_eval*100:>4.1f}%) | {res['pe_5plus']:>2} ({res['pe_5plus']/n_eval*100:>3.1f}%)")
print("="*110)

