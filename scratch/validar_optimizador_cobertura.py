import csv
import itertools
import numpy as np
from collections import defaultdict

def compute_cooccurrence_matrix(filepath='historico_quini_completo.csv'):
    matrix = np.zeros((46, 46))
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nums = [int(row[f'B{i}']) for i in range(1, 7)]
                for u, v in itertools.combinations(nums, 2):
                    matrix[u, v] += 1
                    matrix[v, u] += 1
                count += 1
            except Exception:
                continue
    if count > 0:
        matrix /= count
    return matrix

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

def prob_hits_poisson_binomial_flat(p):
    p0, p1, p2, p3, p4, p5 = p
    dp1_0 = 1.0 - p0
    dp1_1 = p0
    
    dp2_0 = dp1_0 * (1.0 - p1)
    dp2_1 = dp1_1 * (1.0 - p1) + dp1_0 * p1
    dp2_2 = dp1_1 * p1
    
    dp3_0 = dp2_0 * (1.0 - p2)
    dp3_1 = dp2_1 * (1.0 - p2) + dp2_0 * p2
    dp3_2 = dp2_2 * (1.0 - p2) + dp2_1 * p2
    dp3_3 = dp2_2 * p2
    
    dp4_0 = dp3_0 * (1.0 - p3)
    dp4_1 = dp3_1 * (1.0 - p3) + dp3_0 * p3
    dp4_2 = dp3_2 * (1.0 - p3) + dp3_1 * p3
    dp4_3 = dp3_3 * (1.0 - p3) + dp3_2 * p3
    dp4_4 = dp3_3 * p3
    
    dp5_0 = dp4_0 * (1.0 - p4)
    dp5_1 = dp4_1 * (1.0 - p4) + dp4_0 * p4
    dp5_2 = dp4_2 * (1.0 - p4) + dp4_1 * p4
    dp5_3 = dp4_3 * (1.0 - p4) + dp4_2 * p4
    dp5_4 = dp4_4 * (1.0 - p4) + dp4_3 * p4
    dp5_5 = dp4_4 * p4
    
    q5 = 1.0 - p5
    h3 = dp5_3 * q5 + dp5_2 * p5
    h4 = dp5_4 * q5 + dp5_3 * p5
    h5 = dp5_5 * q5 + dp5_4 * p5
    h6 = dp5_5 * p5
    
    return h3, h4, h5, h6

def select_final_3_tickets(boletos_viables, borda_scores, cooccur_matrix, gamma=0.75, w_co=250.0):
    # 1. Ticket A: Borda Core Spike (Highest Borda sum + Pair Co-occurrence)
    ticket_A = max(boletos_viables, key=lambda b: sum(borda_scores[n] for n in b) + w_co * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)))
    
    # Apply discount to Borda scores for selected numbers in A
    borda_B = borda_scores.copy()
    for n in ticket_A:
        borda_B[n] *= gamma
        
    # 2. Ticket B: Affinity Clique (Pure High Co-occurrence Pair Density + anchor on remaining high Borda)
    ticket_B = max(boletos_viables, key=lambda b: sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)) * 500.0 + sum(borda_B[n] for n in b))
    
    # Apply discount for B
    borda_C = borda_B.copy()
    for n in ticket_B:
        borda_C[n] *= gamma
        
    # 3. Ticket C: Hypergraph Coverage (Maximize remaining Borda sum + pair co-occurrence)
    ticket_C = max(boletos_viables, key=lambda b: sum(borda_C[n] for n in b) + (w_co * 0.8) * sum(cooccur_matrix[u, v] for u, v in itertools.combinations(b, 2)))
    
    return [ticket_A, ticket_B, ticket_C]

def main():
    print("--- VALIDANDO SELECCIÓN DE 3 TICKETS (CORE SPIKE + AFFINITY CLIQUE + HYPERGRAPH COVER) ---")
    cooccur = compute_cooccurrence_matrix('historico_quini_completo.csv')
    
    # Top 15 pool predicted for Sorteo 3408
    top_pool = [4, 5, 6, 9, 12, 17, 21, 23, 25, 26, 31, 39, 43, 44, 45]
    winning_segunda = {4, 20, 25, 31, 39, 42}
    
    borda_scores = {n: 0.0 for n in range(46)}
    for rank, num in enumerate(top_pool):
        borda_scores[num] = 45.0 - rank
        
    todas_comb = list(itertools.combinations(top_pool, 6))
    boletos_viables = [c for c in todas_comb if es_ticket_valido(c)]
    
    tickets_finales = select_final_3_tickets(boletos_viables, borda_scores, cooccur, gamma=0.75, w_co=250.0)
    
    max_individual_hits = 0
    total_unique_hits = set()
    names = ["SUPER TICKET A (CORE SPIKE)", "SUPER TICKET B (AFFINITY CLIQUE)", "SUPER TICKET C (HYPERGRAPH COVER)"]
    
    for idx, t in enumerate(tickets_finales):
        matched = sorted(list(set(t).intersection(winning_segunda)))
        hits = len(matched)
        max_individual_hits = max(max_individual_hits, hits)
        total_unique_hits.update(matched)
        print(f"\n{names[idx]}:")
        print(f"  Números: {sorted(t)}")
        print(f"  Aciertos La Segunda ({hits}): {matched}")
        
    print("\n==================================================")
    print(f"MÁXIMO ACIERTO EN UN SOLO TICKET: {max_individual_hits} aciertos!")
    print(f"TOTAL ACIERTOS COBIERTOS ENTRE LOS 3 TICKETS: {len(total_unique_hits)} aciertos {sorted(list(total_unique_hits))}")
    print("==================================================")

if __name__ == '__main__':
    main()
