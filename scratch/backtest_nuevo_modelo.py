import csv
import itertools
import numpy as np
import os
import sys
from collections import defaultdict

def cargar_historico(filepath='historico_quini_limpio.csv'):
    if not os.path.exists(filepath):
        filepath = 'c:/Users/Administrador/Desktop/lot/historico_quini_limpio.csv'
    draws = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                draws.append({
                    'sorteo': int(row['Sorteo']),
                    'fecha': row['Fecha'],
                    'numbers': sorted([int(row[f'B{i}']) for i in range(1, 7)])
                })
            except (ValueError, KeyError):
                continue
    draws.reverse() # Chronological (oldest first, newest last)
    return draws

def es_ticket_valido(ticket):
    ticket = sorted(ticket)
    gaps_le_1 = sum(1 for i in range(1, len(ticket)) if ticket[i] - ticket[i-1] <= 1)
    if gaps_le_1 > 2:
        return False
    spread = ticket[5] - ticket[0]
    if spread < 15 or spread > 45:
        return False
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(c > 4 for c in decenas.values()):
        return False
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares == 0 or pares == 6:
        return False
    return True

def compute_cooccurrence_matrix_up_to(draws, up_to_idx):
    matrix = np.zeros((46, 46))
    sub_draws = draws[:up_to_idx]
    total = len(sub_draws) or 1
    for d in sub_draws:
        nums = d['numbers']
        for u, v in itertools.combinations(nums, 2):
            matrix[u, v] += 1.0
            matrix[v, u] += 1.0
    return matrix / total

def compute_borda_scores_up_to(draws, up_to_idx, top_k=15):
    # Calculate frequency over last 20 and 50 draws + geometric delay
    sub = draws[:up_to_idx]
    n_history = len(sub)
    
    freq_20 = Counter()
    for d in sub[-20:]:
        for n in d['numbers']:
            freq_20[n] += 1
            
    freq_50 = Counter()
    for d in sub[-50:]:
        for n in d['numbers']:
            freq_50[n] += 1
            
    last_seen = {}
    for idx, d in enumerate(sub):
        for n in d['numbers']:
            last_seen[n] = idx
            
    scores = {}
    for n in range(46):
        delay = n_history - 1 - last_seen.get(n, 0)
        # Score = weighted combination of short-term freq, mid-term freq, and recency delay decay
        scores[n] = (freq_20[n] * 2.0) + freq_50[n] + np.exp(-delay / 8.0) * 3.0
        
    borda_ranking = sorted(range(46), key=lambda x: scores[x], reverse=True)
    borda_dict = {n: 45 - rank for rank, n in enumerate(borda_ranking)}
    return borda_dict, borda_ranking[:top_k]

def select_old_tickets(boletos_viables, borda_scores):
    tickets_old = []
    curr_b = borda_scores.copy()
    for _ in range(3):
        best_t = max(boletos_viables, key=lambda b: sum(curr_b[n] for n in b))
        tickets_old.append(best_t)
        for n in best_t:
            curr_b[n] *= 0.0
    return tickets_old

def select_new_tickets(boletos_viables, borda_scores, cooccur_matrix, gamma=0.75, w_co=250.0):
    ticket_A = max(boletos_viables, key=lambda b: sum(borda_scores[n] for n in b) + w_co * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)))
    
    borda_B = borda_scores.copy()
    for n in ticket_A:
        borda_B[n] *= gamma
        
    ticket_B = max(boletos_viables, key=lambda b: sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda_B[n] for n in b))
    
    borda_C = borda_B.copy()
    for n in ticket_B:
        borda_C[n] *= gamma
        
    ticket_C = max(boletos_viables, key=lambda b: sum(borda_C[n] for n in b) + (w_co * 0.8) * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)))
    
    return [ticket_A, ticket_B, ticket_C]

def ejecutar_backtest(eval_window=100):
    print("==================================================")
    print(f"BACKTEST COMPARATIVO: MODELO VIEJO VS MODELO NUEVO")
    print(f"Evaluando los últimos {eval_window} sorteos históricos...")
    print("==================================================\n")
    
    draws = cargar_historico()
    total_draws = len(draws)
    if total_draws < eval_window + 50:
        eval_window = total_draws - 50
        
    start_idx = total_draws - eval_window
    
    hits_old_dist = defaultdict(int)
    hits_new_dist = defaultdict(int)
    
    max_hits_old_per_draw = []
    max_hits_new_per_draw = []
    
    pool_hits_list = []
    
    for t_idx in range(start_idx, total_draws):
        target_draw = set(draws[t_idx]['numbers'])
        
        # 1. Compute Borda Scores & Co-occurrence up to t_idx
        borda_scores, top_pool = compute_borda_scores_up_to(draws, t_idx, top_k=15)
        cooccur_matrix = compute_cooccurrence_matrix_up_to(draws, t_idx)
        
        pool_hits = len(set(top_pool).intersection(target_draw))
        pool_hits_list.append(pool_hits)
        
        # 2. Candidate tickets in pool
        todas_comb = list(itertools.combinations(top_pool, 6))
        boletos_viables = [c for c in todas_comb if es_ticket_valido(c)]
        if not boletos_viables:
            boletos_viables = todas_comb[:100]
            
        # 3. Model Old (gamma=0.0, disjoint)
        tickets_old = select_old_tickets(boletos_viables, borda_scores)
        max_old = 0
        for b in tickets_old:
            h = len(set(b).intersection(target_draw))
            hits_old_dist[h] += 1
            max_old = max(max_old, h)
        max_hits_old_per_draw.append(max_old)
        
        # 4. Model New (Pair Co-occurrence + Tiered Clique + gamma=0.75)
        tickets_new = select_new_tickets(boletos_viables, borda_scores, cooccur_matrix, gamma=0.75, w_co=250.0)
        max_new = 0
        for b in tickets_new:
            h = len(set(b).intersection(target_draw))
            hits_new_dist[h] += 1
            max_new = max(max_new, h)
        max_hits_new_per_draw.append(max_new)
        
    print("==================================================")
    print(f"RESULTADOS DEL BACKTEST ({eval_window} SORTEOS EVALUADOS)")
    print("==================================================")
    
    print("\n1. DISTRIBUCIÓN TOTAL DE ACIERTOS POR TICKET:")
    print("Aciertos | Modelo Viejo (Tickets) | Modelo Nuevo (Tickets) | Dif. Puntos %")
    print("---------+-----------------------+-----------------------+--------------")
    tot_tickets = eval_window * 3
    for h in range(7):
        n_old = hits_old_dist[h]
        n_new = hits_new_dist[h]
        pct_old = (n_old / tot_tickets) * 100
        pct_new = (n_new / tot_tickets) * 100
        diff = pct_new - pct_old
        sign = "+" if diff > 0 else ""
        print(f" {h} Hits  | {n_old:4d} ({pct_old:6.2f}%)       | {n_new:4d} ({pct_new:6.2f}%)       | {sign}{diff:+.2f}%")
        
    print("---------+-----------------------+-----------------------+--------------")
    old_3plus = sum(hits_old_dist[h] for h in range(3, 7))
    new_3plus = sum(hits_new_dist[h] for h in range(3, 7))
    pct_old_3plus = (old_3plus / tot_tickets) * 100
    pct_new_3plus = (new_3plus / tot_tickets) * 100
    print(f" 3+ Hits | {old_3plus:4d} ({pct_old_3plus:6.2f}%)       | {new_3plus:4d} ({pct_new_3plus:6.2f}%)       | {pct_new_3plus - pct_old_3plus:+.2f}%")
    
    print("\n2. MÁXIMO ACIERTO ALCANZADO EN UN SOLO TICKET POR SORTEO:")
    old_draws_3plus = sum(1 for m in max_hits_old_per_draw if m >= 3)
    new_draws_3plus = sum(1 for m in max_hits_new_per_draw if m >= 3)
    old_draws_4plus = sum(1 for m in max_hits_old_per_draw if m >= 4)
    new_draws_4plus = sum(1 for m in max_hits_new_per_draw if m >= 4)
    
    print(f"- Sorteos con al menos 1 ticket de 3+ aciertos: Modelo Viejo: {old_draws_3plus}/{eval_window} ({(old_draws_3plus/eval_window)*100:.1f}%) vs Modelo Nuevo: {new_draws_3plus}/{eval_window} ({(new_draws_3plus/eval_window)*100:.1f}%)")
    print(f"- Sorteos con al menos 1 ticket de 4+ aciertos: Modelo Viejo: {old_draws_4plus}/{eval_window} ({(old_draws_4plus/eval_window)*100:.1f}%) vs Modelo Nuevo: {new_draws_4plus}/{eval_window} ({(new_draws_4plus/eval_window)*100:.1f}%)")
    
    prom_hits_old = sum(h * c for h, c in hits_old_dist.items()) / tot_tickets
    prom_hits_new = sum(h * c for h, c in hits_new_dist.items()) / tot_tickets
    print(f"\n- Promedio de aciertos por ticket: Modelo Viejo: {prom_hits_old:.4f} vs Modelo Nuevo: {prom_hits_new:.4f} (Mejora: {((prom_hits_new - prom_hits_old)/prom_hits_old)*100:+.2f}%)")
    print("==================================================")

if __name__ == '__main__':
    from collections import Counter
    ejecutar_backtest(eval_window=150)
