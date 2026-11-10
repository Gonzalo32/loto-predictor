import os, sys, itertools, csv
from collections import defaultdict, Counter
import numpy as np
import time

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_adaptive_pool,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_attention_mlp_scores
)
from cobertura_combinatoria import select_optimal_wheel_triplet, select_balanced_bibd_triplet

# Lógica del modelo original inicial (Pool 12, Anti-SPOF, Overlap <= 3, Forzar 12 nums)
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

def run_extended_backtest():
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    
    # Tomamos los últimos 100 sorteos (~1 año completo de sorteos de Quini 6)
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-100:]
    n_eval = len(eval_draws)
    
    print("="*100)
    print(f" EXTENDED WALK-FORWARD BACKTEST EN {n_eval} SORTEOS HISTÓRICOS (~1 AÑO COMPLETO)")
    print(f" Sorteo #{eval_draws[0]['sorteo']} ({eval_draws[0]['fecha']}) al #{eval_draws[-1]['sorteo']} ({eval_draws[-1]['fecha']})")
    print(" Comparativa: Modelo Original vs Modelo BIBD Estático vs Modelo BIBD Dinámico por Régimen vs Azar")
    print("="*100)
    
    models = {
        '1. Modelo Original (Pool 12, Anti-SPOF, Overlap <= 3)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'pe_5+': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        },
        '2. Modelo BIBD Estático (Pool 9 Compacto + Trío Núcleo)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'pe_5+': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        },
        '3. Modelo BIBD Dinámico (Régimen Momentum vs Reversión)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'pe_5+': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        },
        '4. Control Aleatorio (Azar Puro 3 Boletos)': {
            'trad_3+': 0, 'trad_4+': 0, 'trad_5+': 0, 'any_4+': 0, 'pe_5+': 0,
            'max_trad': [], 'desparramados': 0, 'boletos_3': 0, 'boletos_4': 0
        }
    }
    
    start_time = time.time()
    
    for step_i, target_draw in enumerate(eval_draws):
        target_sorteo = target_draw['sorteo']
        group = [d for d in draws if d['sorteo'] == target_sorteo]
        real_by_mod = {d['modalidad']: set(d['numbers']) for d in group}
        target_trad = real_by_mod.get('Tradicional', set())
        target_seg = real_by_mod.get('Segunda', set())
        target_rev = real_by_mod.get('Revancha', set())
        target_ss = real_by_mod.get('SiempreSale', set())
        pozo_extra = target_trad | target_seg | target_rev
        
        train_draws = [d for d in draws if d['sorteo'] < target_sorteo]
        
        # Extracción de características base
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']: counts_60[n] += 1
            
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']: recency_scores[n] += weight
            
        pcr_scores, cooccur_exp_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)
        attn_scores = compute_attention_mlp_scores(train_draws)
        
        # Borda Estático
        borda_stat = defaultdict(float)
        models_stat = [
            (counts_60, 1.5), (recency_scores, 2.0), (pcr_scores, 1.4), (ddm_scores, 1.8),
            ({n: attn_scores[n] for n in range(46)}, 1.5)
        ]
        for m_scores, w in models_stat:
            ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
            for rank, n in enumerate(ranked):
                borda_stat[n] += w * (45 - rank)
        ranking_stat = sorted(range(46), key=lambda x: borda_stat[x], reverse=True)
        
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        # 1. GENERAR BOLETOS MODELO ORIGINAL (Pool 12)
        pool_12 = sorted(ranking_stat[:12])
        combos_12 = [c for c in itertools.combinations(pool_12, 6) if es_ticket_valido(c, recent_numbers)] or list(itertools.combinations(pool_12, 6))[:50]
        t_orig = old_select_triplet(combos_12, pool_12, borda_stat, cooccur_exp_mat)
        
        # 2. GENERAR BOLETOS MODELO BIBD ESTÁTICO (Pool 9)
        pool_9_stat = sorted(ranking_stat[:9])
        combos_9_stat = list(itertools.combinations(pool_9_stat, 6))
        viable_9_stat = [c for c in combos_9_stat if es_ticket_valido(c, recent_numbers)] or combos_9_stat
        t_bibd_stat = select_optimal_wheel_triplet(viable_9_stat, pool_9_stat, borda_stat, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # 3. GENERAR BOLETOS MODELO BIBD DINÁMICO POR RÉGIMEN
        last_5 = train_draws[-5:]
        n_repeats = 0
        for i in range(1, len(last_5)):
            n_repeats += len(set(last_5[i]['numbers']) & set(last_5[i-1]['numbers']))
        avg_repeats = n_repeats / 4.0
        
        if avg_repeats >= 1.25:
            w_rec, w_ddm = 3.0, 1.0
        elif avg_repeats <= 0.75:
            w_rec, w_ddm = 1.0, 3.0
        else:
            w_rec, w_ddm = 2.0, 1.8
            
        borda_dyn = defaultdict(float)
        models_dyn = [
            (counts_60, 1.5), (recency_scores, w_rec), (pcr_scores, 1.4), (ddm_scores, w_ddm),
            ({n: attn_scores[n] for n in range(46)}, 1.5)
        ]
        for m_scores, w in models_dyn:
            ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
            for rank, n in enumerate(ranked):
                borda_dyn[n] += w * (45 - rank)
        ranking_dyn = sorted(range(46), key=lambda x: borda_dyn[x], reverse=True)
        pool_9_dyn = sorted(ranking_dyn[:9])
        combos_9_dyn = list(itertools.combinations(pool_9_dyn, 6))
        viable_9_dyn = [c for c in combos_9_dyn if es_ticket_valido(c, recent_numbers)] or combos_9_dyn
        t_bibd_dyn = select_optimal_wheel_triplet(viable_9_dyn, pool_9_dyn, borda_dyn, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # 4. CONTROL ALEATORIO (Azar Puro)
        np.random.seed(target_sorteo)
        t_azar = [tuple(sorted(np.random.choice(range(46), 6, replace=False))) for _ in range(3)]
        
        strats = [
            ('1. Modelo Original (Pool 12, Anti-SPOF, Overlap <= 3)', t_orig, pool_12),
            ('2. Modelo BIBD Estático (Pool 9 Compacto + Trío Núcleo)', t_bibd_stat, pool_9_stat),
            ('3. Modelo BIBD Dinámico (Régimen Momentum vs Reversión)', t_bibd_dyn, pool_9_dyn),
            ('4. Control Aleatorio (Azar Puro 3 Boletos)', t_azar, [])
        ]
        
        for name, t_list, p_list in strats:
            p_hits = len(set(p_list) & target_trad) if p_list else 0
            hits_trad = [len(set(t) & target_trad) for t in t_list]
            max_h_trad = max(hits_trad)
            max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
            max_pe = max(len(set(t) & pozo_extra) for t in t_list)
            
            # Desparramo: pool >= 3 pero max ticket < 3
            if p_hits >= 3 and max_h_trad < 3:
                models[name]['desparramados'] += 1
                
            models[name]['max_trad'].append(max_h_trad)
            models[name]['boletos_3'] += sum(1 for h in hits_trad if h == 3)
            models[name]['boletos_4'] += sum(1 for h in hits_trad if h >= 4)
            
            if max_h_trad >= 3: models[name]['trad_3+'] += 1
            if max_h_trad >= 4: models[name]['trad_4+'] += 1
            if max_h_trad >= 5: models[name]['trad_5+'] += 1
            if max_any >= 4: models[name]['any_4+'] += 1
            if max_pe >= 5: models[name]['pe_5+'] += 1

    elapsed = time.time() - start_time
    print(f"\n[OK] Simulación de {n_eval} sorteos finalizada en {elapsed:.1f} segundos.\n")
    
    print("="*105)
    print(f"{'Modelo Evaluado (100 Sorteos)':<55} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*105)
    for name, m in models.items():
        avg_t = np.mean(m['max_trad'])
        p3 = (m['trad_3+'] / n_eval) * 100
        n4_t = m['trad_4+']
        p4_t = (n4_t / n_eval) * 100
        n4_any = m['any_4+']
        p4_any = (n4_any / n_eval) * 100
        pe5 = m['pe_5+']
        desp = m['desparramados']
        b3 = m['boletos_3']
        print(f"{name:<55} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_t} ({p4_t:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*105)
    
    print("\n" + "="*85)
    print(" DETALLE ADICIONAL DE EFICIENCIA EN 100 SORTEOS:")
    print("="*85)
    for name, m in models.items():
        if 'Aleatorio' in name: continue
        print(f" * {name}:")
        print(f"    - Boletos individuales con 3 aciertos: {m['boletos_3']} boletos")
        print(f"    - Boletos individuales con 4+ aciertos: {m['boletos_4']} boletos")
        print(f"    - Sorteos con desparramo (Pool >= 3 pero Boletos < 3): {m['desparramados']} fechas")
        print(f"    - Sorteos con 5+ aciertos en Pozo Extra: {m['pe_5+']} fechas")
    print("="*85)

if __name__ == '__main__':
    run_extended_backtest()

