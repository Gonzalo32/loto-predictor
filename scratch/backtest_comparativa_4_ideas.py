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

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]

# -------------------------------------------------------------
# Filtro Cuantitativo Estricto de Cuantiles 95% (Idea 4)
# -------------------------------------------------------------
def es_ticket_cuantil_estricto(ticket, recent_numbers=None):
    if not es_ticket_valido(ticket, recent_numbers):
        return False
    s = sum(ticket)
    # Suma entre 90 y 178 (95% empírico del Quini 6)
    if s < 90 or s > 178:
        return False
    # Spread entre 22 y 43
    spread = ticket[5] - ticket[0]
    if spread < 22 or spread > 43:
        return False
    # Paridades equilibradas: 2, 3 o 4 pares (96% empírico)
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares not in (2, 3, 4):
        return False
    return True

# -------------------------------------------------------------
# Función auxiliar de BIBD Clique Core (Idea 1)
# -------------------------------------------------------------
def select_bibd_clique(pool_9, borda, cooccur_matrix, boletos_viables):
    all_trios = list(itertools.combinations(pool_9, 3))
    def score_trio(trio):
        u, v, w = trio
        return (borda.get(u,0)+borda.get(v,0)+borda.get(w,0))*0.4 + (cooccur_matrix[u,v]+cooccur_matrix[u,w]+cooccur_matrix[v,w])*400.0
    core = list(max(all_trios, key=score_trio))
    rest = [n for n in pool_9 if n not in core]
    core_pairs = [(core[0], core[1]), (core[0], core[2]), (core[1], core[2])]
    pairs_of_rest = list(itertools.combinations(rest, 2))
    partitions = []
    for p1 in pairs_of_rest:
        rem1 = [n for n in rest if n not in p1]
        for p2 in itertools.combinations(rem1, 2):
            p3 = tuple(sorted([n for n in rem1 if n not in p2]))
            part = tuple(sorted([p1, p2, p3]))
            if part not in partitions: partitions.append(part)
    best_triplet = None
    best_score = -float('inf')
    viable_set = set(tuple(sorted(c)) for c in boletos_viables) if boletos_viables else None
    for part in partitions:
        for perm in itertools.permutations(part):
            t1 = tuple(sorted(list(core_pairs[0]) + [n for n in rest if n not in perm[0]]))
            if viable_set and t1 not in viable_set: continue
            t2 = tuple(sorted(list(core_pairs[1]) + [n for n in rest if n not in perm[1]]))
            if viable_set and t2 not in viable_set: continue
            t3 = tuple(sorted(list(core_pairs[2]) + [n for n in rest if n not in perm[2]]))
            if viable_set and t3 not in viable_set: continue
            total_score = sum(sum(borda.get(n, 0) for n in t) + 350.0 * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(t, 2)) for t in [t1, t2, t3])
            if total_score > best_score:
                best_score = total_score
                best_triplet = [t1, t2, t3]
    return best_triplet or select_optimal_wheel_triplet(boletos_viables or [], pool_9, borda, cooccur_matrix)

def run_comparative_benchmark():
    print("="*95)
    print(" BENCHMARK RIGUROSO DE 4 IDEAS DE MEJORA ESTADÍSTICA (50 SORTEOS WALK-FORWARD)")
    print(f" Periodo: Sorteo #{eval_draws[0]['sorteo']} al #{eval_draws[-1]['sorteo']}")
    print("="*95)
    
    metrics = {
        '0. Baseline Actual (BIBD Pool 9 Tradicional)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []},
        '1. Idea 1: Clique Core (Máxima Sinergia de Pares)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []},
        '2. Idea 2: Ensamble Dinámico por Régimen (Momentum vs Reversión)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []},
        '3. Idea 3: Especialización Multi-Modal (T1 Trad, T2 Segunda, T3 SiempreSale)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []},
        '4. Idea 4: Filtros Cuantitativos 95% (Suma 90-178, Spread 22-43, Paridades)': {'trad_3+': 0, 'trad_4+': 0, 'any_4+': 0, 'pe_5+': 0, 'max_trad': []},
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
        
        # Borda Base Estándar
        borda_base = defaultdict(float)
        models_base = [
            (counts_60, 1.5), (recency_scores, 2.0), (pcr_scores, 1.4), (ddm_scores, 1.8),
            ({n: attn_scores[n] for n in range(46)}, 1.5)
        ]
        for m_scores, w in models_base:
            ranked = sorted(range(46), key=lambda x: m_scores[x], reverse=True)
            for rank, n in enumerate(ranked):
                borda_base[n] += w * (45 - rank)
        ranking_base = sorted(range(46), key=lambda x: borda_base[x], reverse=True)
        pool_9_base = sorted(ranking_base[:9])
        
        recent_numbers = set()
        for d in train_draws[-5:]: recent_numbers.update(d['numbers'])
        
        combos_base = list(itertools.combinations(pool_9_base, 6))
        viable_base = [c for c in combos_base if es_ticket_valido(c, recent_numbers)] or combos_base
        
        # -------------------------------------------------------------
        # TICKET GENERATION: STRATEGY 0 (BASELINE)
        # -------------------------------------------------------------
        t_base = select_optimal_wheel_triplet(viable_base, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # -------------------------------------------------------------
        # STRATEGY 1 (IDEA 1: CLIQUE CORE)
        # -------------------------------------------------------------
        t_clique = select_bibd_clique(pool_9_base, borda_base, cooccur_exp_mat, viable_base)
        
        # -------------------------------------------------------------
        # STRATEGY 2 (IDEA 2: REGIME-AWARE DYNAMIC ENSEMBLE)
        # Analizar últimos 5 sorteos: tasa de repetición vs tasa de atraso
        # -------------------------------------------------------------
        last_5 = train_draws[-5:]
        n_repeats = 0
        for i in range(1, len(last_5)):
            n_repeats += len(set(last_5[i]['numbers']) & set(last_5[i-1]['numbers']))
        avg_repeats = n_repeats / 4.0
        
        # Si promedio de repeticiones > 1.25 -> régimen caliente (momentum)
        # Si promedio < 0.75 -> régimen frío (reversión / ddm)
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
        combos_dyn = list(itertools.combinations(pool_9_dyn, 6))
        viable_dyn = [c for c in combos_dyn if es_ticket_valido(c, recent_numbers)] or combos_dyn
        t_dyn = select_optimal_wheel_triplet(viable_dyn, pool_9_dyn, borda_dyn, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # -------------------------------------------------------------
        # STRATEGY 3 (IDEA 3: MULTI-MODAL SPECIALIZED TRIPLET)
        # T1: Mejor combinación de Tradicional
        # T2: Mejor combinación de Segunda Vuelta (entrenado con transiciones de Segunda)
        # T3: Mejor combinación de Siempre Sale / Transición cruzada
        # -------------------------------------------------------------
        # Sub-sorteos por modalidad
        draws_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional']
        draws_seg  = [d for d in train_draws if d['modalidad'] == 'Segunda']
        draws_ss   = [d for d in train_draws if d['modalidad'] == 'SiempreSale']
        
        counts_seg = Counter()
        for d in draws_seg[-30:]:
            for n in d['numbers']: counts_seg[n] += 1
            
        counts_ss = Counter()
        for d in draws_ss[-30:]:
            for n in d['numbers']: counts_ss[n] += 1
            
        # T1 = Mejor boleto de viable_base para Tradicional
        t1_mm = max(viable_base, key=lambda c: sum(borda_base[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        
        # T2 = Especialista Segunda: bonificación por afinidad de Segunda Vuelta
        pool_10_seg = sorted(range(46), key=lambda n: borda_base[n] + 18.0 * counts_seg[n], reverse=True)[:9]
        combos_seg = list(itertools.combinations(pool_10_seg, 6))
        viable_seg = [c for c in combos_seg if es_ticket_valido(c, recent_numbers)] or combos_seg
        t2_mm = max(viable_seg, key=lambda c: sum(borda_base[n] + 18.0 * counts_seg[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        
        # T3 = Especialista Siempre Sale / Pozo Extra: bonificación por Siempre Sale
        pool_10_ss = sorted(range(46), key=lambda n: borda_base[n] + 18.0 * counts_ss[n], reverse=True)[:9]
        combos_ss = list(itertools.combinations(pool_10_ss, 6))
        viable_ss = [c for c in combos_ss if es_ticket_valido(c, recent_numbers)] or combos_ss
        t3_mm = max(viable_ss, key=lambda c: sum(borda_base[n] + 18.0 * counts_ss[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)))
        t_mm = [t1_mm, t2_mm, t3_mm]
        
        # -------------------------------------------------------------
        # STRATEGY 4 (IDEA 4: CUANTILES ESTRICTOS 95% CI)
        # Aplica filtros de suma 90-178, spread 22-43, y paridad 2,3,4
        # -------------------------------------------------------------
        viable_quant = [c for c in combos_base if es_ticket_cuantil_estricto(c, recent_numbers)]
        if len(viable_quant) < 3:
            viable_quant = [c for c in combos_base if es_ticket_valido(c, recent_numbers)] or combos_base
        t_quant = select_optimal_wheel_triplet(viable_quant, pool_9_base, borda_base, cooccur_exp_mat, max_overlap=4, w_cooccur=350.0)
        
        # EVALUACIÓN DE LAS 5 ESTRATEGIAS
        evaluated_pairs = [
            ('0. Baseline Actual (BIBD Pool 9 Tradicional)', t_base),
            ('1. Idea 1: Clique Core (Máxima Sinergia de Pares)', t_clique),
            ('2. Idea 2: Ensamble Dinámico por Régimen (Momentum vs Reversión)', t_dyn),
            ('3. Idea 3: Especialización Multi-Modal (T1 Trad, T2 Segunda, T3 SiempreSale)', t_mm),
            ('4. Idea 4: Filtros Cuantitativos 95% (Suma 90-178, Spread 22-43, Paridades)', t_quant)
        ]
        
        for name, t_list in evaluated_pairs:
            hits_trad = [len(set(t) & target_trad) for t in t_list]
            max_h_trad = max(hits_trad)
            max_any = max(max(len(set(t) & m) for t in t_list) for m in [target_trad, target_seg, target_rev, target_ss] if m)
            max_pe = max(len(set(t) & pozo_extra) for t in t_list)
            
            metrics[name]['max_trad'].append(max_h_trad)
            if max_h_trad >= 3: metrics[name]['trad_3+'] += 1
            if max_h_trad >= 4: metrics[name]['trad_4+'] += 1
            if max_any >= 4: metrics[name]['any_4+'] += 1
            if max_pe >= 5: metrics[name]['pe_5+'] += 1

    elapsed = time.time() - start_time
    print(f"\n[OK] Benchmark completado en {elapsed:.1f} segundos.\n")
    
    print("="*105)
    print(f"{'Estrategia / Idea':<52} | {'Prom. Trad.':<11} | {'3+ Trad %':<10} | {'4+ Trad (Premios)':<18} | {'4+ Cualquier Mod.'}")
    print("="*105)
    
    total_s = len(eval_draws)
    for name, m in metrics.items():
        avg_t = np.mean(m['max_trad'])
        p3 = (m['trad_3+'] / total_s) * 100
        n4_trad = m['trad_4+']
        p4_trad = (n4_trad / total_s) * 100
        n4_any = m['any_4+']
        p4_any = (n4_any / total_s) * 100
        pe5 = m['pe_5+']
        print(f"{name:<52} | {avg_t:<11.2f} | {p3:<9.1f}% | {n4_trad} ({p4_trad:.1f}%)          | {n4_any} ({p4_any:.1f}%) [PE 5+: {pe5}]")
    print("="*105)

if __name__ == '__main__':
    run_comparative_benchmark()

