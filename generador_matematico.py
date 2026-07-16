import csv
import os
import sys
import random
from collections import defaultdict

def load_data(filepath="historico_quini_completo.csv"):
    draws = []
    if not os.path.exists(filepath):
        filepath = "c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv"
        
    if not os.path.exists(filepath):
        print("Error: No se encontró historico_quini_completo.csv")
        return []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sorteo = int(row['Sorteo'])
                fecha = row['Fecha']
                modalidad = row['Modalidad'].strip()
                numbers = sorted([int(row[f'B{i}']) for i in range(1, 7)])
                draws.append({
                    'sorteo': sorteo,
                    'fecha': fecha,
                    'modalidad': modalidad,
                    'numbers': numbers
                })
            except Exception:
                continue
                
    modality_order = {'Tradicional': 1, 'Segunda': 2, 'Revancha': 3, 'SiempreSale': 4}
    draws.sort(key=lambda x: (x['sorteo'], modality_order.get(x['modalidad'], 9)))
    return draws

def es_ticket_valido_matematico(ticket, recent_numbers, last_draw_numbers, hot_numbers):
    ticket = sorted(ticket)
    
    # 1. Paridad: entre 2 y 4 números pares (evita extremos raros de 0 o 6 pares)
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares < 2 or pares > 4:
        return False
        
    # 2. Consecutividad: máximo una pareja consecutiva
    gaps_le_1 = sum(1 for i in range(1, len(ticket)) if ticket[i] - ticket[i-1] <= 1)
    if gaps_le_1 > 1:
        return False
        
    # 3. Spread (dispersión): entre 18 y 44
    spread = ticket[5] - ticket[0]
    if spread < 18 or spread > 44:
        return False
        
    # 4. Decenas: máximo 3 números por decena (evita acumulaciones inusuales)
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(count > 3 for count in decenas.values()):
        return False
        
    # 5. Overlap con el último sorteo: debe compartir EXACTAMENTE 1 o 2 números
    # (Esto ocurre en el 56.8% de los sorteos, mientras que 3+ ocurre en < 2.2%)
    overlap = len(set(ticket).intersection(last_draw_numbers))
    if overlap < 1 or overlap > 2:
        return False
        
    # 6. Partición Caliente/Frío: debe tener exactamente 4 números calientes (delay <= 7)
    # y 2 números fríos (delay > 7). Esto cumple con el valor esperado de la curva geométrica.
    hot_count = sum(1 for n in ticket if n in hot_numbers)
    if hot_count != 4:
        return False
        
    return True

def generar_sugerencias():
    draws = load_data()
    if not draws:
        return
        
    n_total = len(draws)
    ultimo_sorteo = draws[-1]
    
    print("==================================================")
    print("GENERADOR MATEMÁTICO DE BOLETOS - QUINI 6")
    print("==================================================")
    print(f"Último sorteo registrado: Nro {ultimo_sorteo['sorteo']} ({ultimo_sorteo['fecha']})")
    print(f"Modalidad: {ultimo_sorteo['modalidad']} | Números: {list(ultimo_sorteo['numbers'])}")
    print("--------------------------------------------------")
    
    # Calculate current delays
    last_seen = {n: -1 for n in range(46)}
    for idx, d in enumerate(draws):
        for n in d['numbers']:
            last_seen[n] = idx
            
    current_delays = {}
    for n in range(46):
        current_delays[n] = n_total - 1 - last_seen[n] if last_seen[n] != -1 else n_total
        
    # Calientes (delay <= 7) vs Fríos (delay > 7)
    hot_numbers = [n for n in range(46) if current_delays[n] <= 7]
    cold_numbers = [n for n in range(46) if current_delays[n] > 7]
    
    print(f"Números calientes actuales (delay <= 7): {hot_numbers}")
    print(f"Números fríos actuales (delay > 7):     {cold_numbers}")
    print("--------------------------------------------------")
    
    last_draw_numbers = set(ultimo_sorteo['numbers'])
    
    sugerencias = []
    intentos = 0
    max_intentos = 1000000
    
    # We want 3 unique valid tickets
    while len(sugerencias) < 3 and intentos < max_intentos:
        intentos += 1
        # Select 4 hot numbers and 2 cold numbers
        selected_hot = random.sample(hot_numbers, 4)
        selected_cold = random.sample(cold_numbers, 2)
        candidate = sorted([int(x) for x in list(selected_hot) + list(selected_cold)])
        
        if es_ticket_valido_matematico(candidate, None, last_draw_numbers, set(hot_numbers)):
            tup = tuple(candidate)
            if tup not in sugerencias:
                sugerencias.append(tup)
                
    if len(sugerencias) < 3:
        print("No se pudieron generar suficientes sugerencias bajo los estrictos filtros de probabilidad.")
    else:
        print("\nBoletos sugeridos matemáticamente para el próximo sorteo:")
        print("(Cumplen con filtros de decenas, consecutividad, spread, paridad, delay y overlap modal)")
        print("--------------------------------------------------")
        for idx, ticket in enumerate(sugerencias):
            # Print details of the ticket
            overlap_last = len(set(ticket).intersection(last_draw_numbers))
            delays_str = ", ".join(f"{n}(d={current_delays[n]})" for n in ticket)
            print(f"Boleto {idx+1}: {list(ticket)}")
            print(f"  - Delays actuales: {delays_str}")
            print(f"  - Coincidencia con último sorteo: {overlap_last} números")
            print("--------------------------------------------------")
            
if __name__ == "__main__":
    generar_sugerencias()
