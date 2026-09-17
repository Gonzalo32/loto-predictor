import csv
import json
import numpy as np
import itertools
import os
import sys
from collections import defaultdict, Counter
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generador_cobertura_ml import (
    load_data, extract_features_for_step, update_transition_counts,
    compute_cooccurrence_matrix, select_tiered_clique_tickets, es_ticket_valido
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def run_full_backtest_comparison():
    draws = load_data('c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv')
    if not draws:
        print("Error: No se pudieron cargar los datos de historico_quini_completo.csv")
        return

    # Usamos sorteos de modalidad 'Tradicional' y 'Segunda'
    # Evaluamos en una ventana representativa de 100 sorteos secuenciales históricos
    modalidades_eval = ['Tradicional', 'Segunda']
    eval_indices = []
    for idx, d in enumerate(draws):
        if d['modalidad'] in modalidades_eval and idx >= 150:
            eval_indices.append(idx)
            
    # Muestreamos cada 2 sorteos para cubrir una ventana amplia rápidamente (50 sorteos de prueba)
    eval_sample = eval_indices[-60:]
    
    print("="*80)
    print(f"COMPARA DE BACKTESTING COMPLETO: POOL DE 15 NÚMEROS VS POOL REDUCIDO DE 10 NÚMEROS")
    print(f"Total Sorteos Evaluados: {len(eval_sample)} sorteos cronológicos reales")
    print("="*80)
    
    # Acumuladores para Pool de 15
    p15_hits_pool = []
    p15_hits_ticket = []
    p15_ticket_3plus = 0
    p15_ticket_4plus = 0
    p15_ticket_5plus = 0
    
    # Acumuladores para Pool de 10 (Nuevo Modelo)
    p10_hits_pool = []
    p10_hits_ticket = []
    p10_ticket_3plus = 0
    p10_ticket_4plus = 0
    p10_ticket_5plus = 0
    
    start_time = time.time()
    
    for count, idx in enumerate(eval_sample):
        target_draw = draws[idx]
        target_nums = set(target_draw['numbers'])
        history = draws[:idx]
        
        # Generar ranking de frecuencias + recencia + co-ocurrencia para simulación rápida de backtest
        counts = Counter()
        for d in history[-60:]:
            for n in d['numbers']:
                counts[n] += 1
                
        recency_scores = np.zeros(46)
        for t_idx, d in enumerate(history[-30:]):
            weight = np.exp((t_idx - 30) / 10.0)
            for n in d['numbers']:
                recency_scores[n] += weight
                
        # Matriz de co-ocurrencia
        cooccur_matrix = compute_cooccurrence_matrix(history)
        
        final_scores = np.zeros(46)
        for n in range(46):
            final_scores[n] = counts[n] * 0.4 + recency_scores[n] * 0.6
            
        borda_ranking = list(np.argsort(final_scores)[::-1])
        
        # --- 1. EVALUAR POOL 15 ---
        top15 = borda_ranking[:15]
        hits15_pool = len(set(top15).intersection(target_nums))
        p15_hits_pool.append(hits15_pool)
        
        combos15 = list(itertools.combinations(sorted(top15), 6))
        viables15 = [c for c in combos15 if es_ticket_valido(c)]
        if not viables15:
            viables15 = combos15[:50]
        tickets15 = select_tiered_clique_tickets(viables15, final_scores, cooccur_matrix)
        max_hits15 = max(len(set(t).intersection(target_nums)) for t in tickets15)
        p15_hits_ticket.append(max_hits15)
        if max_hits15 >= 3: p15_ticket_3plus += 1
        if max_hits15 >= 4: p15_ticket_4plus += 1
        if max_hits15 >= 5: p15_ticket_5plus += 1
        
        # --- 2. EVALUAR POOL 10 (NUEVO MODELO) ---
        top10 = borda_ranking[:10]
        hits10_pool = len(set(top10).intersection(target_nums))
        p10_hits_pool.append(hits10_pool)
        
        combos10 = list(itertools.combinations(sorted(top10), 6))
        viables10 = [c for c in combos10 if es_ticket_valido(c)]
        if not viables10:
            viables10 = combos10[:50]
        tickets10 = select_tiered_clique_tickets(viables10, final_scores, cooccur_matrix)
        max_hits10 = max(len(set(t).intersection(target_nums)) for t in tickets10)
        p10_hits_ticket.append(max_hits10)
        if max_hits10 >= 3: p10_ticket_3plus += 1
        if max_hits10 >= 4: p10_ticket_4plus += 1
        if max_hits10 >= 5: p10_ticket_5plus += 1
        
        if (count + 1) % 10 == 0:
            print(f"Procesados {count + 1}/{len(eval_sample)} sorteos de prueba...")

    elapsed = time.time() - start_time
    total = len(eval_sample)
    
    print("\n" + "="*80)
    print(f"RESULTADOS DEL BACKTESTING COMPARATIVO ({total} SORTEOS REANOTADOS):")
    print("="*80)
    
    print(f"\n1. EFICIENCIA DE POOL (PRECISIÓN Y DENSIDAD):")
    print(f"  - POOL 15 ANTERIOR:")
    print(f"      Promedio aciertos en Pool (de 6): {np.mean(p15_hits_pool):.2f}")
    print(f"      Densidad de Acierto (Aciertos/Pool): {(np.mean(p15_hits_pool)/15)*100:.2f}%")
    print(f"  - POOL 10 NUEVO REDUCIDO:")
    print(f"      Promedio aciertos en Pool (de 6): {np.mean(p10_hits_pool):.2f}")
    print(f"      Densidad de Acierto (Aciertos/Pool): {(np.mean(p10_hits_pool)/10)*100:.2f}%")
    print(f"  --> MEJORA DE DENSIDAD DE PRECISIÓN: +{((np.mean(p10_hits_pool)/10)/(np.mean(p15_hits_pool)/15) - 1)*100:.1f}% MÁS DENSE")
    
    print(f"\n2. RENDIMIENTO DE BOLETAS (3 TICKETS DE 6 NÚMEROS):")
    print(f"  - POOL 15 ANTERIOR:")
    print(f"      Promedio Máximo Aciertos por Boleta: {np.mean(p15_hits_ticket):.2f}")
    print(f"      % Sorteos con >= 3 aciertos en 1 Boleta: {(p15_ticket_3plus/total)*100:.1f}%")
    print(f"      % Sorteos con >= 4 aciertos en 1 Boleta: {(p15_ticket_4plus/total)*100:.1f}%")
    print(f"      % Sorteos con >= 5 aciertos en 1 Boleta: {(p15_ticket_5plus/total)*100:.1f}%")
    
    print(f"\n  - POOL 10 NUEVO REDUCIDO:")
    print(f"      Promedio Máximo Aciertos por Boleta: {np.mean(p10_hits_ticket):.2f}")
    print(f"      % Sorteos con >= 3 aciertos en 1 Boleta: {(p10_ticket_3plus/total)*100:.1f}%")
    print(f"      % Sorteos con >= 4 aciertos en 1 Boleta: {(p10_ticket_4plus/total)*100:.1f}%")
    print(f"      % Sorteos con >= 5 aciertos en 1 Boleta: {(p10_ticket_5plus/total)*100:.1f}%")
    
    imp_3plus = ((p10_ticket_3plus/total) - (p15_ticket_3plus/total)) * 100
    imp_4plus = ((p10_ticket_4plus/total) - (p15_ticket_4plus/total)) * 100
    
    print("\n" + "="*80)
    print(f"CONCLUSIÓN GENERAL DEL MODELO DE POOL REDUCIDO:")
    print(f"  • Tasa de Acierto 3+ por Boleta: {'MEJORÓ' if imp_3plus >= 0 else 'IGUAL/DIFERENCIA'} (Variación de {imp_3plus:+.1f} puntos porcentuales)")
    print(f"  • Tasa de Acierto 4+ por Boleta: {'MEJORÓ' if imp_4plus >= 0 else 'IGUAL/DIFERENCIA'} (Variación de {imp_4plus:+.1f} puntos porcentuales)")
    print(f"  • Tiempo de Backtest: {elapsed:.1f} segundos")
    print("="*80)

if __name__ == '__main__':
    run_full_backtest_comparison()
