import csv
import json
import itertools
import numpy as np
from collections import defaultdict, Counter
import time
import os
import sys

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, es_ticket_valido, compute_adaptive_pool,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    compute_attention_mlp_scores
)
from cobertura_combinatoria import select_optimal_wheel_triplet

def run_production_backtest():
    draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
    eval_draws = [d for d in draws if d['modalidad'] == 'Tradicional'][-50:]
    
    print("="*95)
    print(" BACKTEST DE CÓDIGO EN PRODUCCIÓN: EVALUANDO NUEVO MOTOR BIBD Y DENSIDAD COMPACTA")
    print(f" Periodo: Sorteos #{eval_draws[0]['sorteo']} ({eval_draws[0]['fecha']}) al #{eval_draws[-1]['sorteo']} ({eval_draws[-1]['fecha']})")
    print("="*95)
    
    hits_3_plus = 0
    hits_4_plus = 0
    hits_5_plus = 0
    any_4_plus = 0
    pe_5_plus = 0
    
    max_hits_list = []
    pool_hits_list = []
    desparramados = 0
    
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
        
        # Ensamble de scores idéntico a producción
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
                
        # Usamos exactamente las funciones de producción
        top_pool = compute_adaptive_pool(borda, alpha=0.20, min_pool=9, max_pool=9)
        
        recent_numbers = set()
        for d in train_draws[-5:]:
            recent_numbers.update(d['numbers'])
            
        combos_pool = list(itertools.combinations(sorted(top_pool), 6))
        boletos_viables = [c for c in combos_pool if es_ticket_valido(c, recent_numbers)]
        if not boletos_viables:
            boletos_viables = [c for c in combos_pool if es_ticket_valido(c)]
        if not boletos_viables:
            boletos_viables = combos_pool[:100]
            
        tickets_produccion = select_optimal_wheel_triplet(
            boletos_viables=boletos_viables,
            pool=sorted(top_pool),
            scores=borda,
            cooccur_matrix=cooccur_exp_mat,
            max_overlap=4,
            w_cooccur=350.0
        )
        
        pool_hits = len(set(top_pool) & target_trad)
        hits_per_ticket = [len(set(t) & target_trad) for t in tickets_produccion]
        max_h = max(hits_per_ticket)
        
        max_h_any = max(
            max(len(set(t) & m) for t in tickets_produccion)
            for m in [target_trad, target_seg, target_rev, target_ss] if m
        )
        max_h_pe = max(len(set(t) & pozo_extra) for t in tickets_produccion)
        
        if pool_hits >= 3 and max_h < 3:
            desparramados += 1
            
        max_hits_list.append(max_h)
        pool_hits_list.append(pool_hits)
        
        if max_h >= 3: hits_3_plus += 1
        if max_h >= 4: hits_4_plus += 1
        if max_h >= 5: hits_5_plus += 1
        if max_h_any >= 4: any_4_plus += 1
        if max_h_pe >= 5: pe_5_plus += 1

    elapsed = time.time() - start_time
    n_total = len(eval_draws)
    
    print(f"\n[OK] Simulación walk-forward completada en {elapsed:.1f} segundos.\n")
    print("="*80)
    print(" RESULTADOS OFICIALES DEL NUEVO MOTOR EN PRODUCCIÓN:")
    print("="*80)
    print(f" * Total sorteos evaluados:              {n_total}")
    print(f" * Promedio de aciertos máximo:          {np.mean(max_hits_list):.2f}")
    print(f" * Sorteos con 3+ aciertos (Tradicional): {hits_3_plus} de {n_total} ({(hits_3_plus/n_total)*100:.1f}%) [Antes: 8.0%]")
    print(f" * Sorteos con 4+ aciertos (Cualquier):   {any_4_plus} de {n_total} ({(any_4_plus/n_total)*100:.1f}%) [Antes: 0.0%]")
    print(f" * Sorteos con 5+ aciertos (Pozo Extra):  {pe_5_plus} de {n_total} ({(pe_5_plus/n_total)*100:.1f}%) [Antes: 3]")
    print(f" * Sorteos con desparramo (Pool>=3 pero Boletos<3): {desparramados} sorteos [Antes: 7 sorteos]")
    print("="*80)

if __name__ == '__main__':
    run_production_backtest()
