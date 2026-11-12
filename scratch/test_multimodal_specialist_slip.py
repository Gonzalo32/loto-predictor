import os, sys, itertools, time
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, compute_antagonist_pairs, es_ticket_valido,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    select_optimal_wheel_triplet
)
from scratch.compare_600_draws_varE import select_pool, es_ticket_valido_enhanced

def compute_fast_attn_by_mod(train_draws, target_mod='Tradicional'):
    mod_draws = [d['numbers'] for d in train_draws if d['modalidad'] == target_mod]
    if len(mod_draws) < 20:
        mod_draws = [d['numbers'] for d in train_draws]
    T = len(mod_draws)
    scores = np.zeros(46)
    sub = mod_draws[-35:]
    T_sub = len(sub)
    for idx, nums in enumerate(sub):
        w = np.exp((idx - T_sub) / 10.0)
        for n in nums:
            scores[n] += w
    return scores

def run_experiment(n_eval=300):
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-n_eval:]
    
    results = {
        'Baseline (B2 Solo Día)': {
            'trad_3': 0, 'trad_4': 0, 'seg_3': 0, 'seg_4': 0, 'rev_4': 0, 'ss_4': 0, 'global_4plus': 0, 'global_3plus': 0, 'pe_4plus': 0, 'pe_5plus': 0
        },
        'Multimodal Especializado (B1=Trad, B2=Segunda+Día, B3=PozoExtra)': {
            'trad_3': 0, 'trad_4': 0, 'seg_3': 0, 'seg_4': 0, 'rev_4': 0, 'ss_4': 0, 'global_4plus': 0, 'global_3plus': 0, 'pe_4plus': 0, 'pe_5plus': 0
        }
    }
    
    t0 = time.time()
    print(f"Evaluando Slip Monomodal vs Multimodal Especializado en {n_eval} sorteos...")
    
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
            
        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)
        
        # Attention por modalidad
        attn_trad = compute_fast_attn_by_mod(train_draws, 'Tradicional')
        attn_seg  = compute_fast_attn_by_mod(train_draws, 'Segunda')
        
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
        
        # Borda Tradicional Base
        borda_base = defaultdict(float)
        for m, w in [(counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_trad[n] for n in range(46)}, 1.5)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda_base[n] += w * (45 - rank)
        for n in range(46):
            if res_counts[n] >= 2: borda_base[n] += 18.0 * (res_counts[n] - 1)
            elif res_counts[n] == 1: borda_base[n] += 4.0
            if n in prev_ss_nums: borda_base[n] += 4.0
            
        # Variantes para B2:
        # 1. Base: borda_day = borda + 16 * counts_day
        borda_day_base = defaultdict(float)
        borda_hib_base = defaultdict(float)
        for n in range(46):
            borda_day_base[n] = borda_base[n] + 16.0 * counts_day[n]
            borda_hib_base[n] = borda_base[n] + 8.0 * counts_day[n]
            
        # 2. Multimodal Especializado: B2 incorpora el Attention Model de La Segunda
        borda_seg_spec = defaultdict(float)
        for m, w in [(counts_60, 1.2), (recency_scores, w_rec), (pcr_scores, 1.2), (ddm_scores, w_ddm), ({n: attn_seg[n] for n in range(46)}, 2.2)]:
            rk = sorted(range(46), key=lambda x: m[x], reverse=True)
            for rank, n in enumerate(rk): borda_seg_spec[n] += w * (45 - rank)
        for n in range(46):
            borda_seg_spec[n] += 14.0 * counts_day[n]
            if res_counts[n] >= 2: borda_seg_spec[n] += 18.0 * (res_counts[n] - 1)
            elif res_counts[n] == 1: borda_seg_spec[n] += 4.0
            if n in prev_ss_nums: borda_seg_spec[n] += 4.0

        p_base = select_pool(borda_base, saturated_decades, pool_size=9)
        def get_v(p):
            c_list = list(itertools.combinations(p, 6))
            v = [c for c in c_list if es_ticket_valido_enhanced(c, recent_numbers, saturated_decades, antag, min_distinct_decades=3, filter_sum=True, filter_hilow=True)]
            if len(v) < 3: v = [c for c in c_list if es_ticket_valido(c, recent_numbers, saturated_decades, None, min_distinct_decades=3)]
            return v or c_list[:10]
            
        v_base = get_v(p_base)
        b1 = select_optimal_wheel_triplet(v_base, p_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)[0]
        
        # Evaluar Config 1: Baseline
        p_day_b = select_pool(borda_day_base, saturated_decades, pool_size=9)
        p_hib_b = select_pool(borda_hib_base, saturated_decades, pool_size=10)
        v_day_b = get_v(p_day_b)
        v_hib_b = get_v(p_hib_b)
        cands_b2_b = [c for c in v_day_b if 2 <= len(set(c) & set(b1)) <= 4] or [c for c in v_day_b if c != b1]
        b2_b = max(cands_b2_b, key=lambda c: sum(borda_day_base[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        q12_b = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2_b, 4))
        cands_b3_b = [c for c in v_hib_b if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2_b)) <= 4) and len(set(b1) & set(b2_b) & set(c)) <= 2] or [c for c in v_hib_b if c != b1 and c != b2_b]
        b3_b = max(cands_b3_b, key=lambda c: sum(borda_hib_base[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12_b))
        tickets_base = [b1, b2_b, b3_b]
        
        # Evaluar Config 2: Multimodal Especializado
        p_seg = select_pool(borda_seg_spec, saturated_decades, pool_size=9)
        p_hib_spec = select_pool(borda_hib_base, saturated_decades, pool_size=10)
        v_seg = get_v(p_seg)
        v_hib_spec = get_v(p_hib_spec)
        cands_b2_s = [c for c in v_seg if 2 <= len(set(c) & set(b1)) <= 4] or [c for c in v_seg if c != b1]
        b2_s = max(cands_b2_s, key=lambda c: sum(borda_seg_spec[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        q12_s = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2_s, 4))
        cands_b3_s = [c for c in v_hib_spec if (2 <= len(set(c) & set(b1)) <= 4) and (2 <= len(set(c) & set(b2_s)) <= 4) and len(set(b1) & set(b2_s) & set(c)) <= 2] or [c for c in v_hib_spec if c != b1 and c != b2_s]
        b3_s = max(cands_b3_s, key=lambda c: sum(borda_hib_base[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * len(set(itertools.combinations(c, 4)) - q12_s))
        tickets_spec = [b1, b2_s, b3_s]
        
        for name, tickets in [('Baseline (B2 Solo Día)', tickets_base), ('Multimodal Especializado (B1=Trad, B2=Segunda+Día, B3=PozoExtra)', tickets_spec)]:
            h_trad = max(len(set(t) & target_trad) for t in tickets)
            h_seg  = max(len(set(t) & target_seg) for t in tickets) if target_seg else 0
            h_rev  = max(len(set(t) & target_rev) for t in tickets) if target_rev else 0
            h_ss   = max(len(set(t) & target_ss) for t in tickets) if target_ss else 0
            h_pe   = max(len(set(t) & pozo_extra) for t in tickets)
            
            res = results[name]
            if h_trad >= 3: res['trad_3'] += 1
            if h_trad >= 4: res['trad_4'] += 1
            if h_seg >= 3:  res['seg_3'] += 1
            if h_seg >= 4:  res['seg_4'] += 1
            if h_rev >= 4:  res['rev_4'] += 1
            if h_ss >= 4:   res['ss_4'] += 1
            
            if max(h_trad, h_seg, h_rev, h_ss) >= 4: res['global_4plus'] += 1
            if max(h_trad, h_seg, h_rev, h_ss) >= 3: res['global_3plus'] += 1
            if h_pe >= 4: res['pe_4plus'] += 1
            if h_pe >= 5: res['pe_5plus'] += 1

        if (step_i + 1) % 50 == 0:
            print(f"[{step_i + 1}/{n_eval}] completados en {time.time()-t0:.1f}s...")
            
    print("\n" + "="*115)
    print(f"{'ESTRATEGIA':<42} | {'TRAD 4+':<8} | {'SEG 4+':<8} | {'GLOB 4+':<8} | {'GLOB 3+':<8} | {'PE 4+':<8} | {'PE 5+':<8}")
    print("="*115)
    for name, res in results.items():
        print(f"{name:<42} | {res['trad_4']:>2} ({res['trad_4']/n_eval*100:>3.1f}%) | {res['seg_4']:>2} ({res['seg_4']/n_eval*100:>3.1f}%) | {res['global_4plus']:>2} ({res['global_4plus']/n_eval*100:>3.1f}%) | {res['global_3plus']:>2} ({res['global_3plus']/n_eval*100:>4.1f}%) | {res['pe_4plus']:>2} ({res['pe_4plus']/n_eval*100:>4.1f}%) | {res['pe_5plus']:>2} ({res['pe_5plus']/n_eval*100:>3.1f}%)")
    print("="*115)

if __name__ == '__main__':
    run_experiment(300)
