"""
Test de Mejora: Filtro y Ponderacion de Rebote Cruzado Inmediato (Cross-Modal Rebound)
Evaluando 500 sorteos
"""
import os, sys, itertools
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import time

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores
)
from scratch.test_hybrid_option_a_and_b import compute_fast_attention_mlp_scores, select_optimal_wheel_triplet

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-500:]
n_eval = len(eval_draws)

strategies = {
    'Portfolio Especializado Actual': {
        'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []
    },
    'Portfolio + Rebote Cruzado Inmediato (1-3 numeros del sorteo previo)': {
        'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []
    },
    'Portfolio + Boost Rebote Cruzado (Segunda/SS a Tradicional)': {
        'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []
    },
    'Portfolio + Filtro Estricto 4 Decenas + Rebote Cruzado': {
        'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []
    }
}

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
    prev_sorteo = max(d['sorteo'] for d in train_draws)
    prev_group = [d for d in train_draws if d['sorteo'] == prev_sorteo]
    prev_all = set()
    for d in prev_group: prev_all.update(d['numbers'])
    
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
    
    last_5 = train_draws[-5:]
    n_repeats = sum(len(set(last_5[i]['numbers']) & set(last_5[i-1]['numbers'])) for i in range(1, len(last_5)))
    avg_repeats = n_repeats / 4.0
    if avg_repeats >= 1.25: w_rec, w_ddm = 3.0, 1.0
    elif avg_repeats <= 0.75: w_rec, w_ddm = 1.0, 3.0
    else: w_rec, w_ddm = 2.0, 1.8
        
    recent_numbers = set()
    for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
    
    # Borda Base (Tradicional)
    borda_base = defaultdict(float)
    models_base = [
        (counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm),
        ({n: attn_scores[n] for n in range(46)}, 1.5)
    ]
    for m_scores, w in models_base:
        ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
        for rank, n in enumerate(ranked): borda_base[n] += w * (45 - rank)
            
    # Borda Dia
    same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
    borda_day = defaultdict(float)
    for n in range(46): borda_day[n] = borda_base[n] + 16.0 * counts_day[n]
        
    # Borda Boost Rebote Cruzado
    borda_boost = defaultdict(float)
    for n in range(46):
        boost = 12.0 if n in prev_all else 0.0
        borda_boost[n] = borda_day[n] + boost
        
    # Pool 9 Base
    ranking_base = sorted(range(46), key=lambda x: borda_base[x], reverse=True)
    pool_9_base = sorted(ranking_base[:9])
    viable_base = [c for c in itertools.combinations(pool_9_base, 6) if es_ticket_valido(c, recent_numbers)] or list(itertools.combinations(pool_9_base, 6))
    t_base = select_optimal_wheel_triplet(viable_base, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    
    # Pool 9 Day
    ranking_day = sorted(range(46), key=lambda x: borda_day[x], reverse=True)
    pool_9_day = sorted(ranking_day[:9])
    viable_day = [c for c in itertools.combinations(pool_9_day, 6) if es_ticket_valido(c, recent_numbers)] or list(itertools.combinations(pool_9_day, 6))
    t_day = select_optimal_wheel_triplet(viable_day, pool_9_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    
    # 1. Baseline Portfolio
    t_portfolio_actual = [t_base[0], t_day[0] if t_day[0] != t_base[0] else t_day[1], t_base[1]]
    
    # 2. Portfolio con Filtro de Rebote (exigir entre 1 y 3 numeros de prev_all)
    def es_rebote_ok(ticket):
        r = len(set(ticket) & prev_all)
        return 1 <= r <= 3
        
    viable_rebote_base = [c for c in viable_base if es_rebote_ok(c)] or viable_base
    viable_rebote_day = [c for c in viable_day if es_rebote_ok(c)] or viable_day
    t_base_reb = select_optimal_wheel_triplet(viable_rebote_base, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_day_reb = select_optimal_wheel_triplet(viable_rebote_day, pool_9_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_portfolio_reb = [t_base_reb[0], t_day_reb[0] if t_day_reb[0] != t_base_reb[0] else t_day_reb[1], t_base_reb[1]]
    
    # 3. Portfolio con Boost Rebote Cruzado
    ranking_boost = sorted(range(46), key=lambda x: borda_boost[x], reverse=True)
    pool_9_boost = sorted(ranking_boost[:9])
    viable_boost = [c for c in itertools.combinations(pool_9_boost, 6) if es_ticket_valido(c, recent_numbers)] or list(itertools.combinations(pool_9_boost, 6))
    t_boost = select_optimal_wheel_triplet(viable_boost, pool_9_boost, borda_boost, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_portfolio_boost = [t_base[0], t_boost[0] if t_boost[0] != t_base[0] else t_boost[1], t_day[0] if t_day[0] != t_base[0] else t_day[1]]
    
    # 4. Portfolio con Filtro 4 Decenas + Rebote
    def es_4dec_y_rebote(ticket):
        dec = len(set(n // 10 for n in ticket))
        r = len(set(ticket) & prev_all)
        return dec >= 4 and 1 <= r <= 3
        
    v_4d_base = [c for c in viable_base if es_4dec_y_rebote(c)] or viable_rebote_base
    v_4d_day = [c for c in viable_day if es_4dec_y_rebote(c)] or viable_rebote_day
    t_4d_base = select_optimal_wheel_triplet(v_4d_base, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_4d_day = select_optimal_wheel_triplet(v_4d_day, pool_9_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_portfolio_4d = [t_4d_base[0], t_4d_day[0] if t_4d_day[0] != t_4d_base[0] else t_4d_day[1], t_4d_base[1]]
    
    candidates = [
        ('Portfolio Especializado Actual', t_portfolio_actual),
        ('Portfolio + Rebote Cruzado Inmediato (1-3 numeros del sorteo previo)', t_portfolio_reb),
        ('Portfolio + Boost Rebote Cruzado (Segunda/SS a Tradicional)', t_portfolio_boost),
        ('Portfolio + Filtro Estricto 4 Decenas + Rebote Cruzado', t_portfolio_4d)
    ]
    
    for name, t_list in candidates:
        hits_trad = [len(set(t) & target_trad) for t in t_list]
        max_h_trad = max(hits_trad)
        max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
        max_pe = max(len(set(t) & pozo_extra) for t in t_list)
        
        strategies[name]['max_trad'].append(max_h_trad)
        if max_h_trad >= 3: strategies[name]['trad_3+'] += 1
        if max_h_trad >= 4: strategies[name]['trad_4+'] += 1
        if max_any >= 4: strategies[name]['any_4+'] += 1
        if max_pe >= 5: strategies[name]['pe_5+'] += 1

print("="*115)
print(f"{'Estrategia':<65} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
print("="*115)
for name, s in strategies.items():
    avg_t = np.mean(s['max_trad'])
    p3 = (s['trad_3+'] / n_eval) * 100
    n4_t = s['trad_4+']
    p4_t = (n4_t / n_eval) * 100
    n4_any = s['any_4+']
    p4_any = (n4_any / n_eval) * 100
    pe5 = s['pe_5+']
    print(f"{name:<65} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
print("="*115)

