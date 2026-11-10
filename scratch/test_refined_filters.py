"""
Test de Nuevos Filtros Finos:
- Paridad estricta (2, 3 o 4 pares)
- Maximo 2 con misma terminacion (ultimo digito)
- Suma entre 110 y 165
- Maximo 1 numero con atraso > 15
Evaluando 500 sorteos historicos
"""
import os, sys, itertools
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

WORKSPACE_DIR = r'c:\Users\Administrador\Desktop\lot'
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, compute_exponential_cooccurrence_scores,
    compute_delta_delay_momentum_scores, compute_antagonist_pairs,
    select_balanced_pool_9
)
from scratch.test_hybrid_option_a_and_b import compute_fast_attention_mlp_scores, select_optimal_wheel_triplet

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-500:]
n_eval = len(eval_draws)

def es_ticket_valido_refinado(ticket, recent_numbers=None, saturated_decades=None, antagonist_pairs=None, delays_dict=None):
    ticket = sorted(ticket)
    # 1. Consecutividad: max 1 consecutiva
    consec = sum(1 for i in range(len(ticket)-1) if ticket[i+1] == ticket[i]+1)
    if consec > 1: return False
    # 2. Spread: entre 15 y 45
    if ticket[5] - ticket[0] < 15 or ticket[5] - ticket[0] > 45: return False
    # 3. Decenas: max 3 por decena y max 2 en la del 40
    decenas = Counter(n // 10 for n in ticket)
    if max(decenas.values()) > 3 or decenas.get(4, 0) > 2: return False
    if len(decenas) < 4: return False # 4 decenas distintas
    # 4. Paridad Estricta: 2, 3 o 4 pares
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares not in (2, 3, 4): return False
    # 5. Terminaciones: max 2 con mismo ultimo digito
    endings = Counter(n % 10 for n in ticket)
    if max(endings.values()) > 2: return False
    # 6. Suma Gaussiana entre 110 y 165
    s = sum(ticket)
    if s < 105 or s > 165: return False
    # 7. Hot/cold: 1 a 5 recientes
    if recent_numbers:
        r_cnt = sum(1 for n in ticket if n in recent_numbers)
        if r_cnt < 1 or r_cnt > 5: return False
    # 8. Decada saturada
    if saturated_decades:
        for dec in saturated_decades:
            if decenas.get(dec, 0) > 2: return False
    # 9. Antagonicos
    if antagonist_pairs:
        for u, v in itertools.combinations(ticket, 2):
            if tuple(sorted([u, v])) in antagonist_pairs: return False
    # 10. Atraso profundo: max 1 con atraso > 15
    if delays_dict:
        cold_cnt = sum(1 for n in ticket if delays_dict.get(n, 0) > 15)
        if cold_cnt > 1: return False
    return True

print("Iniciando backtest de 500 sorteos comparando Portfolio Actual vs Portfolio Refinado...")

results_actual = {'trad_4': 0, 'any_4': 0, 'pe_5': 0, 'trad_3': 0}
results_refinado = {'trad_4': 0, 'any_4': 0, 'pe_5': 0, 'trad_3': 0}

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
        w = np.exp((t_idx - 30) / 10.0)
        for n in d['numbers']: recency_scores[n] += w
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
    
    # Delays
    last_seen = {n: -1 for n in range(46)}
    for idx_d, d in enumerate(train_draws):
        for n in d['numbers']: last_seen[n] = idx_d
    T_total = len(train_draws)
    delays_dict = {n: (T_total - 1 - last_seen[n]) if last_seen[n] != -1 else T_total for n in range(46)}
    
    # Borda Base
    borda_base = defaultdict(float)
    models_base = [(counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm), ({n: attn_scores[n] for n in range(46)}, 1.5)]
    for m_scores, w in models_base:
        ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
        for rank, n in enumerate(ranked): borda_base[n] += w * (45 - rank)
            
    # Borda Dia
    same_day_draws = [d for d in train_draws if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
    counts_day = Counter(n for d in same_day_draws[-40:] for n in d['numbers'])
    borda_day = defaultdict(float)
    borda_hib = defaultdict(float)
    for n in range(46):
        borda_day[n] = borda_base[n] + 16.0 * counts_day[n]
        borda_hib[n] = borda_base[n] + 8.0 * counts_day[n]
        
    # Saturated Decades
    saturated_decades = set()
    last_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional'][-1]
    for dec, cnt in Counter(n // 10 for n in last_trad['numbers']).items():
        if cnt >= 3: saturated_decades.add(dec)
        
    antag = compute_antagonist_pairs(train_draws)
    
    pool_base = select_balanced_pool_9(borda_base, saturated_decades)
    pool_day = select_balanced_pool_9(borda_day, saturated_decades)
    pool_hib = select_balanced_pool_9(borda_hib, saturated_decades)
    
    # 1. Estrategia Actual
    from generador_cobertura_ml import es_ticket_valido
    v_base_act = [c for c in itertools.combinations(pool_base, 6) if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)] or list(itertools.combinations(pool_base, 6))
    v_day_act = [c for c in itertools.combinations(pool_day, 6) if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)] or list(itertools.combinations(pool_day, 6))
    v_hib_act = [c for c in itertools.combinations(pool_hib, 6) if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)] or list(itertools.combinations(pool_hib, 6))
    
    t_base_act = select_optimal_wheel_triplet(v_base_act, pool_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_day_act = select_optimal_wheel_triplet(v_day_act, pool_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_hib_act = select_optimal_wheel_triplet(v_hib_act, pool_hib, borda_hib, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    
    t_actual = [t_base_act[0], t_day_act[0] if t_day_act[0] != t_base_act[0] else t_day_act[1], t_hib_act[0] if (t_hib_act[0] != t_base_act[0] and t_hib_act[0] != t_day_act[0]) else t_base_act[1]]
    
    # 2. Estrategia Refinada (Paridad 2,3,4 + Terminaciones <=2 + Suma 105-165 + Max 1 atrasado)
    def get_v_ref(pool, borda_sc):
        combos = list(itertools.combinations(pool, 6))
        v = [c for c in combos if es_ticket_valido_refinado(c, recent_numbers, saturated_decades, antag, delays_dict)]
        if len(v) < 3:
            v = [c for c in combos if es_ticket_valido(c, recent_numbers, saturated_decades, antag, min_distinct_decades=4)]
        if len(v) < 3:
            v = combos[:10]
        return v
        
    v_base_ref = get_v_ref(pool_base, borda_base)
    v_day_ref = get_v_ref(pool_day, borda_day)
    v_hib_ref = get_v_ref(pool_hib, borda_hib)
    
    t_base_ref = select_optimal_wheel_triplet(v_base_ref, pool_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_day_ref = select_optimal_wheel_triplet(v_day_ref, pool_day, borda_day, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    t_hib_ref = select_optimal_wheel_triplet(v_hib_ref, pool_hib, borda_hib, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    
    t_refinado = [t_base_ref[0], t_day_ref[0] if t_day_ref[0] != t_base_ref[0] else t_day_ref[1], t_hib_ref[0] if (t_hib_ref[0] != t_base_ref[0] and t_hib_ref[0] != t_day_ref[0]) else t_base_ref[1]]
    
    # Evaluar ambos
    for name, t_list, d_res in [('Actual', t_actual, results_actual), ('Refinado', t_refinado, results_refinado)]:
        h_trad = [len(set(t) & target_trad) for t in t_list]
        max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
        max_pe = max(len(set(t) & pozo_extra) for t in t_list)
        if max(h_trad) >= 3: d_res['trad_3'] += 1
        if max(h_trad) >= 4: d_res['trad_4'] += 1
        if max_any >= 4: d_res['any_4'] += 1
        if max_pe >= 5: d_res['pe_5'] += 1

print("\n" + "="*80)
print(f"{'Estrategia Evaluada (500 Sorteos)':<35} | {'3+ Trad %':<10} | {'4+ Trad':<10} | {'4+ Cualquier Mod':<18} | {'Pozo Extra 5+'}")
print("="*80)
for name, d in [('Portfolio Actual', results_actual), ('Portfolio con Filtros Refinados', results_refinado)]:
    p3 = (d['trad_3'] / n_eval) * 100
    n4 = d['trad_4']
    any4 = d['any_4']
    pe5 = d['pe_5']
    print(f"{name:<35} | {p3:<9.1f}% | {n4:<10} | {any4:<18} | {pe5}")
print("="*80)

