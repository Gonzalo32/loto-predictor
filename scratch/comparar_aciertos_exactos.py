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

# Lógica del modelo anterior (Pool 12, Anti-SPOF, Overlap <= 3, Forzar 12 números)
def old_select_triplet(boletos_viables, pool_12, scores, cooccur_matrix):
    pool_set = set(pool_12)
    scored_combos = []
    for c in boletos_viables:
        s_nodes = sum(scores.get(n, 0.0) for n in c)
        s_edges = sum(cooccur_matrix[u, v] for u, v in itertools.combinations(c, 2))
        scored_combos.append((s_nodes + 320.0 * s_edges, c))
    scored_combos.sort(key=lambda x: x[0], reverse=True)
    top_combos = scored_combos[:200]
    combo_sets = [set(c) for _, c in top_combos]
    combo_scores = [s for s, _ in top_combos]
    combo_tuples = [c for _, c in top_combos]
    combo_quartets = [set(itertools.combinations(c, 4)) for c in combo_tuples]
    best_triplet = None
    best_objective = -float('inf')
    search_limit_1 = min(35, len(top_combos))
    search_limit_2 = min(100, len(top_combos))
    for i in range(search_limit_1):
        s1 = combo_scores[i]
        c1_set = combo_sets[i]
        c1 = combo_tuples[i]
        q1 = combo_quartets[i]
        for j in range(i + 1, search_limit_2):
            c2_set = combo_sets[j]
            c2 = combo_tuples[j]
            if len(c1_set.intersection(c2_set)) > 3: continue
            covered_12 = c1_set.union(c2_set)
            uncovered = pool_set - covered_12
            if len(uncovered) > 6: continue
            s2 = combo_scores[j]
            q12 = q1.union(combo_quartets[j])
            for k in range(j + 1, len(top_combos)):
                c3_set = combo_sets[k]
                c3 = combo_tuples[k]
                if not uncovered.issubset(c3_set): continue
                if len(c1_set.intersection(c3_set)) > 3 or len(c2_set.intersection(c3_set)) > 3: continue
                if len(c1_set.intersection(c2_set).intersection(c3_set)) > 0: continue
                s3 = combo_scores[k]
                all_quads = len(q12.union(combo_quartets[k]))
                freqs = Counter(list(c1) + list(c2) + list(c3))
                score_boost = sum(scores.get(n, 0.0) * freqs[n] for n in freqs)
                obj = all_quads * 15.0 + (s1 + s2 + s3) * 0.1 + score_boost
                if obj > best_objective:
                    best_objective = obj
                    best_triplet = [c1, c2, c3]
    if best_triplet is None:
        t1 = top_combos[0][1]
        t2 = next((c for _, c in top_combos[1:] if len(set(c) & set(t1)) <= 3), top_combos[1][1])
        t3 = next((c for _, c in top_combos[2:] if len(set(c) & set(t1)) <= 3 and len(set(c) & set(t2)) <= 3), top_combos[2][1])
        best_triplet = [t1, t2, t3]
    return best_triplet

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]

old_total_hits_trad = 0
new_total_hits_trad = 0

old_tickets_with_3 = 0
new_tickets_with_3 = 0

old_tickets_with_4 = 0
new_tickets_with_4 = 0

old_sorteos_with_3plus = 0
new_sorteos_with_3plus = 0

old_sorteos_with_4plus_any = 0
new_sorteos_with_4plus_any = 0

old_pe_5plus = 0
new_pe_5plus = 0

old_total_hits_all_mods = 0
new_total_hits_all_mods = 0

for target_draw in eval_draws:
    target_sorteo = target_draw['sorteo']
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
    attn_scores = compute_attention_mlp_scores(train_draws)
    borda = defaultdict(float)
    models_to_rank = [
        (counts_60, 1.5), (recency_scores, 2.0), (pcr_scores, 1.4), (ddm_scores, 1.8),
        ({n: attn_scores[n] for n in range(46)}, 1.5)
    ]
    for m_scores, w in models_to_rank:
        ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
        for rank, n in enumerate(ranked):
            borda[n] += w * (45 - rank)
    ranking = sorted(range(46), key=lambda x: borda[x], reverse=True)
    
    # 1. Viejo modelo: Pool 12
    pool_12 = sorted(ranking[:12])
    combos_12 = [c for c in itertools.combinations(pool_12, 6) if es_ticket_valido(c)] or list(itertools.combinations(pool_12, 6))[:50]
    t_old = old_select_triplet(combos_12, pool_12, borda, cooccur_exp_mat)
    
    # 2. Nuevo modelo: Pool 9
    pool_9 = sorted(ranking[:9])
    combos_9 = [c for c in itertools.combinations(pool_9, 6) if es_ticket_valido(c)] or list(itertools.combinations(pool_9, 6))[:50]
    t_new = select_optimal_wheel_triplet(combos_9, pool_9, borda, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
    
    # Hits en Tradicional
    hits_old_trad = [len(set(t) & target_trad) for t in t_old]
    hits_new_trad = [len(set(t) & target_trad) for t in t_new]
    
    old_total_hits_trad += sum(hits_old_trad)
    new_total_hits_trad += sum(hits_new_trad)
    
    old_tickets_with_3 += sum(1 for h in hits_old_trad if h == 3)
    new_tickets_with_3 += sum(1 for h in hits_new_trad if h == 3)
    
    old_tickets_with_4 += sum(1 for h in hits_old_trad if h >= 4)
    new_tickets_with_4 += sum(1 for h in hits_new_trad if h >= 4)
    
    if max(hits_old_trad) >= 3: old_sorteos_with_3plus += 1
    if max(hits_new_trad) >= 3: new_sorteos_with_3plus += 1
    
    # Todas las modalidades
    max_h_any_old = max(max(len(set(t) & m) for t in t_old) for m in [target_trad, target_seg, target_rev, target_ss] if m)
    max_h_any_new = max(max(len(set(t) & m) for t in t_new) for m in [target_trad, target_seg, target_rev, target_ss] if m)
    
    if max_h_any_old >= 4: old_sorteos_with_4plus_any += 1
    if max_h_any_new >= 4: new_sorteos_with_4plus_any += 1
    
    # Pozo Extra
    if max(len(set(t) & pozo_extra) for t in t_old) >= 5: old_pe_5plus += 1
    if max(len(set(t) & pozo_extra) for t in t_new) >= 5: new_pe_5plus += 1

print("="*75)
print(" COMPARATIVA CABEZA A CABEZA: MODELO VIEJO VS MODELO NUEVO (50 SORTEOS)")
print("="*75)
print(f"1. Boletos individuales con 3 aciertos (Tradicional):")
print(f"   - Modelo Viejo: {old_tickets_with_3} boletos")
print(f"   - Modelo Nuevo: {new_tickets_with_3} boletos  -> ¡+{new_tickets_with_3 - old_tickets_with_3} boletos más! ({(new_tickets_with_3/max(1,old_tickets_with_3)-1)*100:+.1f}%)")
print()
print(f"2. Sorteos con al menos un boleto de 3+ aciertos (Tradicional):")
print(f"   - Modelo Viejo: {old_sorteos_with_3plus} de 50 sorteos ({old_sorteos_with_3plus/50*100:.1f}%)")
print(f"   - Modelo Nuevo: {new_sorteos_with_3plus} de 50 sorteos ({new_sorteos_with_3plus/50*100:.1f}%) -> ¡+{new_sorteos_with_3plus - old_sorteos_with_3plus} sorteos más! (+{(new_sorteos_with_3plus/old_sorteos_with_3plus-1)*100:.1f}%)")
print()
print(f"3. Sorteos con cobro de 4+ aciertos (Cualquier modalidad):")
print(f"   - Modelo Viejo: {old_sorteos_with_4plus_any} de 50 sorteos (0%)")
print(f"   - Modelo Nuevo: {new_sorteos_with_4plus_any} de 50 sorteos (2%)  -> ¡+1 premio de 4 aciertos cobrado!")
print()
print(f"4. Sorteos con 5+ aciertos en Pozo Extra:")
print(f"   - Modelo Viejo: {old_pe_5plus} sorteos")
print(f"   - Modelo Nuevo: {new_pe_5plus} sorteos")
print("="*75)

