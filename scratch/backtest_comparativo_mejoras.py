import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
from collections import Counter, defaultdict
import numpy as np
import time

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_exponential_cooccurrence_scores,
    compute_delta_delay_momentum_scores, compute_attention_mlp_scores,
    compute_adaptive_pool, compute_cross_modal_transitions
)
from scratch.test_wheel_algorithm import select_optimal_wheel_triplet

def run_backtest(n_draws_eval=40):
    draws = load_data('historico_quini_completo.csv')
    
    # Obtener los sorteos únicos disponibles
    sorteos_unicos = sorted(list(set(d['sorteo'] for d in draws)))
    eval_sorteos = sorteos_unicos[-n_draws_eval:]
    
    print("="*95)
    print(f" BACKTEST COMPARATIVO: MOTOR ANTERIOR vs MOTOR OPTIMIZADO ({len(eval_sorteos)} SORTEOS)")
    print(f" Rango evaluado: Sorteo #{eval_sorteos[0]} al #{eval_sorteos[-1]}")
    print("="*95)
    
    metrics = {
        'Antiguo (Greedy con Solapamiento 4)': {
            'hits_3_trad': 0, 'hits_4_trad': 0, 'hits_5_trad': 0,
            'hits_3_any': 0,  'hits_4_any': 0,  'hits_5_any': 0,
            'pozo_extra_5plus': 0, 'pozo_extra_6plus': 0,
            'orphans_count': 0, 'total_pool_numbers': 0,
            'max_hit_list': []
        },
        'Nuevo (Rueda Reducida 100% Cobertura + Anti-SPOF)': {
            'hits_3_trad': 0, 'hits_4_trad': 0, 'hits_5_trad': 0,
            'hits_3_any': 0,  'hits_4_any': 0,  'hits_5_any': 0,
            'pozo_extra_5plus': 0, 'pozo_extra_6plus': 0,
            'orphans_count': 0, 'total_pool_numbers': 0,
            'max_hit_list': []
        }
    }
    
    t0 = time.time()
    
    for s_idx, s_num in enumerate(eval_sorteos):
        # Datos reales de este sorteo
        sorteo_rows = [d for d in draws if d['sorteo'] == s_num]
        real_by_mod = {d['modalidad']: set(d['numbers']) for d in sorteo_rows}
        trad_real = real_by_mod.get('Tradicional', set())
        seg_real = real_by_mod.get('Segunda', set())
        rev_real = real_by_mod.get('Revancha', set())
        ss_real = real_by_mod.get('SiempreSale', set())
        pozo_extra_real = trad_real.union(seg_real).union(rev_real)
        
        # Historial de entrenamiento antes de este sorteo
        train_draws = [d for d in draws if d['sorteo'] < s_num]
        if len(train_draws) < 80:
            continue
            
        # Puntuaciones
        counts_60 = Counter()
        for d in train_draws[-60:]:
            for n in d['numbers']:
                counts_60[n] += 1
                
        recency = np.zeros(46)
        for t_idx, d in enumerate(train_draws[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']:
                recency[n] += weight
                
        pcr_scores, cooccur_mat = compute_exponential_cooccurrence_scores(train_draws)
        ddm_scores = compute_delta_delay_momentum_scores(train_draws)
        
        # Cross modal transitions
        cross_mat = compute_cross_modal_transitions(train_draws[-60:])
        last_nums = train_draws[-1]['numbers']
        cross_scores = np.zeros(46)
        for u in last_nums:
            cross_scores += cross_mat[u]
            
        # 1. Ensamble Borda Anterior
        borda_old = defaultdict(float)
        for scores, w in [(counts_60, 1.5), (recency, 2.0), (pcr_scores, 1.4), (ddm_scores, 1.0), (cross_scores, 1.0)]:
            rk = sorted(range(46), key=lambda x: scores[x], reverse=True)
            for rank, n in enumerate(rk):
                borda_old[n] += w * (45 - rank)
        pool_old = compute_adaptive_pool(borda_old, alpha=0.20, min_pool=10, max_pool=12)
        
        # 2. Ensamble Borda Nuevo (DDM con mayor peso para ciclos de maduración)
        borda_new = defaultdict(float)
        for scores, w in [(counts_60, 1.4), (recency, 1.8), (pcr_scores, 1.5), (ddm_scores, 1.8), (cross_scores, 1.2)]:
            rk = sorted(range(46), key=lambda x: scores[x], reverse=True)
            for rank, n in enumerate(rk):
                borda_new[n] += w * (45 - rank)
        pool_new = compute_adaptive_pool(borda_new, alpha=0.20, min_pool=10, max_pool=12)
        
        # --- Generar Boletos Método Antiguo ---
        combos_old = [c for c in itertools.combinations(sorted(pool_old), 6) if es_ticket_valido(c)] or list(itertools.combinations(sorted(pool_old), 6))[:50]
        scored_old = []
        for c in combos_old:
            s_nodes = sum(borda_old[n] for n in c)
            s_edges = sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            scored_old.append((s_nodes + 320.0 * s_edges, c))
        scored_old.sort(key=lambda x: x[0], reverse=True)
        
        tA_old = scored_old[0][1]
        cands_B = [c for _, c in scored_old[1:] if 3 <= len(set(c).intersection(set(tA_old))) <= 4]
        tB_old = cands_B[0] if cands_B else scored_old[1][1]
        
        quads_AB = set(itertools.combinations(tA_old, 4)).union(set(itertools.combinations(tB_old, 4)))
        cands_C = []
        for s_val, c in scored_old:
            if c == tA_old or c == tB_old:
                continue
            new_quads = len(set(itertools.combinations(c, 4)) - quads_AB)
            ovA = len(set(c).intersection(set(tA_old)))
            ovB = len(set(c).intersection(set(tB_old)))
            if 2 <= ovA <= 4 and 2 <= ovB <= 4:
                cands_C.append((new_quads * 75.0 + s_val, c))
        if cands_C:
            cands_C.sort(key=lambda x: x[0], reverse=True)
            tC_old = cands_C[0][1]
        else:
            tC_old = scored_old[2][1]
        triplet_old = [tA_old, tB_old, tC_old]
        
        # --- Generar Boletos Método Nuevo ---
        combos_new = [c for c in itertools.combinations(sorted(pool_new), 6) if es_ticket_valido(c)] or list(itertools.combinations(sorted(pool_new), 6))[:50]
        borda_new_dict = {n: borda_new[n] for n in range(46)}
        triplet_new = select_optimal_wheel_triplet(combos_new, pool_new, borda_new_dict, cooccur_mat, max_overlap=3)
        
        # Evaluar métricas
        for name, triplet, pool_used in [('Antiguo (Greedy con Solapamiento 4)', triplet_old, pool_old),
                                         ('Nuevo (Rueda Reducida 100% Cobertura + Anti-SPOF)', triplet_new, pool_new)]:
            covered_nums = set()
            max_trad = 0
            max_any = 0
            
            for t in triplet:
                t_set = set(t)
                covered_nums.update(t_set)
                
                h_trad = len(t_set.intersection(trad_real))
                h_seg = len(t_set.intersection(seg_real))
                h_rev = len(t_set.intersection(rev_real))
                h_ss = len(t_set.intersection(ss_real))
                
                max_trad = max(max_trad, h_trad)
                max_any = max(max_any, h_trad, h_seg, h_rev, h_ss)
                
                h_pe = len(t_set.intersection(pozo_extra_real))
                if h_pe >= 5: metrics[name]['pozo_extra_5plus'] += 1
                if h_pe >= 6: metrics[name]['pozo_extra_6plus'] += 1
                
            if max_trad >= 3: metrics[name]['hits_3_trad'] += 1
            if max_trad >= 4: metrics[name]['hits_4_trad'] += 1
            if max_trad >= 5: metrics[name]['hits_5_trad'] += 1
            
            if max_any >= 3: metrics[name]['hits_3_any'] += 1
            if max_any >= 4: metrics[name]['hits_4_any'] += 1
            if max_any >= 5: metrics[name]['hits_5_any'] += 1
            
            metrics[name]['max_hit_list'].append(max_trad)
            orphans = set(pool_used) - covered_nums
            metrics[name]['orphans_count'] += len(orphans)
            metrics[name]['total_pool_numbers'] += len(pool_used)
            
        if (s_idx + 1) % 10 == 0:
            print(f" -> Procesados {s_idx + 1}/{len(eval_sorteos)} sorteos...")
            
    elapsed = time.time() - t0
    print(f"\n[OK] Backtest finalizado en {elapsed:.2f} segundos.")
    print("="*95)
    print(f"{'MÉTRICA':<45} | {'MOTOR ANTERIOR':<20} | {'MOTOR NUEVO OPTIMIZADO':<22}")
    print("="*95)
    
    n_total = len(eval_sorteos)
    for k_disp, k_key in [
        ('Sorteos con >= 3 Aciertos en Tradicional', 'hits_3_trad'),
        ('Sorteos con >= 4 Aciertos en Tradicional (PREMIOS)', 'hits_4_trad'),
        ('Sorteos con >= 5 Aciertos en Tradicional (POZO MAYOR)', 'hits_5_trad'),
        ('Sorteos con >= 3 Aciertos en Cualquier Modalidad', 'hits_3_any'),
        ('Sorteos con >= 4 Aciertos en Cualquier Modalidad', 'hits_4_any'),
        ('Boletos con >= 5 Aciertos en Pozo Extra', 'pozo_extra_5plus'),
        ('Boletos con >= 6 Aciertos en Pozo Extra', 'pozo_extra_6plus'),
    ]:
        v_old = metrics['Antiguo (Greedy con Solapamiento 4)'][k_key]
        v_new = metrics['Nuevo (Rueda Reducida 100% Cobertura + Anti-SPOF)'][k_key]
        pct_old = (v_old / n_total) * 100
        pct_new = (v_new / n_total) * 100
        print(f"{k_disp:<45} | {v_old:2d} ({pct_old:5.1f}%)        | {v_new:2d} ({pct_new:5.1f}%)")
        
    print("-"*95)
    # Tasa de utilización del pool (0 huérfanos)
    orph_old = metrics['Antiguo (Greedy con Solapamiento 4)']['orphans_count']
    total_old = metrics['Antiguo (Greedy con Solapamiento 4)']['total_pool_numbers']
    orph_new = metrics['Nuevo (Rueda Reducida 100% Cobertura + Anti-SPOF)']['orphans_count']
    total_new = metrics['Nuevo (Rueda Reducida 100% Cobertura + Anti-SPOF)']['total_pool_numbers']
    
    cov_old = (1.0 - orph_old / total_old) * 100
    cov_new = (1.0 - orph_new / total_new) * 100
    print(f"{'Cobertura Efectiva de Números del Pool':<45} | {cov_old:5.1f}% (Orphans: {orph_old})  | {cov_new:5.1f}% (Orphans: {orph_new})")
    print("="*95)

if __name__ == '__main__':
    run_backtest(40)
