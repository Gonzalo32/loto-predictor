import os, sys, itertools, csv
from collections import defaultdict, Counter
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_adaptive_pool,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_attention_mlp_scores
)
from cobertura_combinatoria import select_optimal_wheel_triplet

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]

results = []
for target_draw in eval_draws:
    target_sorteo = target_draw['sorteo']
    target_fecha = target_draw['fecha']
    
    group = [d for d in draws if d['sorteo'] == target_sorteo]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in group}
    target_trad = real_by_mod.get('Tradicional', set())
    target_seg = real_by_mod.get('Segunda', set())
    target_rev = real_by_mod.get('Revancha', set())
    target_ss = real_by_mod.get('SiempreSale', set())
    
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
    attn_scores = compute_attention_mlp_scores(train_draws)
    
    borda = defaultdict(float)
    models_to_rank = [
        (counts_60, 1.5),
        (recency_scores, 2.0),
        (pcr_scores, 1.4),
        (ddm_scores, 1.8),
        ({n: attn_scores[n] for n in range(46)}, 1.5)
    ]
    for m_scores, w in models_to_rank:
        ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
        for rank, n in enumerate(ranked):
            borda[n] += w * (45 - rank)
            
    top_pool = compute_adaptive_pool(borda, alpha=0.20, min_pool=9, max_pool=9)
    recent_numbers = set()
    for d in train_draws[-5:]:
        recent_numbers.update(d['numbers'])
        
    combos_pool = list(itertools.combinations(sorted(top_pool), 6))
    boletos_viables = [c for c in combos_pool if es_ticket_valido(c, recent_numbers)] or combos_pool
    
    tickets = select_optimal_wheel_triplet(boletos_viables, sorted(top_pool), borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    
    hits_trad = [len(set(t) & target_trad) for t in tickets]
    hits_seg = [len(set(t) & target_seg) for t in tickets]
    hits_rev = [len(set(t) & target_rev) for t in tickets]
    hits_ss = [len(set(t) & target_ss) for t in tickets]
    
    results.append({
        'sorteo': target_sorteo,
        'fecha': target_fecha,
        'pool': sorted(top_pool),
        'trad_sorteo': sorted(list(target_trad)),
        'tickets': tickets,
        'hits_trad': hits_trad,
        'hits_seg': hits_seg,
        'hits_rev': hits_rev,
        'hits_ss': hits_ss
    })

print("=== DETALLE DE SORTEOS DESTACADOS EN EL BACKTEST (50 SORTEOS) ===")
for r in results:
    max_t = max(r['hits_trad'])
    max_s = max(r['hits_seg'])
    max_r = max(r['hits_rev'])
    max_ss = max(r['hits_ss'])
    max_any = max(max_t, max_s, max_r, max_ss)
    if max_any >= 3 or max_t >= 3:
        s_num = r['sorteo']
        s_date = r['fecha']
        print(f"Sorteo #{s_num} ({s_date}) | Tradicional: {r['hits_trad']} | Segunda: {r['hits_seg']} | Revancha: {r['hits_rev']} | SiempreSale: {r['hits_ss']}")
        if max_t >= 4 or max_any >= 4:
            print(f"   >>> PREMIO! Acierto de {max_any} números! Boletos: {r['tickets']}")
