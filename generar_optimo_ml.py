import csv
import numpy as np
from collections import defaultdict
import itertools
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

def leer_datos(archivo='c:/Users/Administrador/Desktop/lot/historico_quini_limpio.csv'):
    datos = []
    if not os.path.exists(archivo):
        return []
    with open(archivo, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bolillas = [int(row[f'B{i}']) for i in range(1, 7)]
                datos.append(bolillas)
            except ValueError:
                continue
    datos.reverse()
    return datos

def es_ticket_perfecto(ticket):
    ticket = sorted(ticket)
    for i in range(1, len(ticket)):
        if ticket[i] - ticket[i-1] <= 1:
            return False
    spread = ticket[5] - ticket[0]
    if spread < 20 or spread > 42:
        return False
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(count > 2 for count in decenas.values()):
        return False
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares == 0 or pares == 6:
        return False
    return True

def seleccionar_mmr(boletos_puntuados, penalizacion_compartido):
    seleccionados = []
    restantes = list(boletos_puntuados)
    for _ in range(3):
        if not restantes: break
        mejor_temp = None
        mejor_score_temp = -99999.0
        for item in restantes:
            ticket, original_score = item
            max_compartidos = 0
            for sel_ticket in seleccionados:
                compartidos = len(set(ticket).intersection(set(sel_ticket)))
                if compartidos > max_compartidos:
                    max_compartidos = compartidos
            score_final = original_score - (max_compartidos * penalizacion_compartido)
            if score_final > mejor_score_temp:
                mejor_score_temp = score_final
                mejor_temp = item
        if mejor_temp:
            seleccionados.append(mejor_temp[0])
            restantes.remove(mejor_temp)
    return seleccionados

def generar_proximos():
    datos = leer_datos()
    if not datos:
        print("Error al leer datos.")
        return
        
    print(f"Entrenando clasificador ML con {len(datos)} sorteos...")
    
    # 1. Construir dataset de entrenamiento
    X_train = []
    y_train = []
    
    for t in range(50, len(datos)):
        hist_t = datos[:t]
        real_t = set(datos[t])
        
        last_seen = {}
        for idx_h, s_h in enumerate(hist_t):
            for n in s_h:
                last_seen[n] = idx_h
                
        cooc = defaultdict(lambda: defaultdict(int))
        for s_h in hist_t:
            for n1, n2 in itertools.combinations(s_h, 2):
                cooc[n1][n2] += 1
                cooc[n2][n1] += 1
                
        cooc_flat = {}
        for n1 in cooc:
            for n2 in cooc[n1]:
                cooc_flat[(n1, n2)] = cooc[n1][n2]
        max_cooc = max(cooc_flat.values()) if cooc_flat else 1
        ultimo_sorteo = hist_t[-1]
        
        for n in range(46):
            delay = t - 1 - last_seen.get(n, -1)
            freq_5 = sum(1 for s_h in hist_t[-5:] if n in s_h) / 5.0
            freq_15 = sum(1 for s_h in hist_t[-15:] if n in s_h) / 15.0
            freq_30 = sum(1 for s_h in hist_t[-30:] if n in s_h) / 30.0
            cooc_last = sum(cooc[n].get(u, 0) for u in ultimo_sorteo) / (6.0 * max_cooc)
            
            X_train.append([delay, freq_5, freq_15, freq_30, cooc_last])
            y_train.append(1 if n in real_t else 0)
            
    # Entrenar RandomForest
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 2. Predecir probabilidades para el proximo sorteo (i = len(datos))
    X_test = []
    last_seen_i = {}
    for idx_h, s_h in enumerate(datos):
        for n in s_h:
            last_seen_i[n] = idx_h
            
    cooc_i = defaultdict(lambda: defaultdict(int))
    for s_h in datos:
        for n1, n2 in itertools.combinations(s_h, 2):
            cooc_i[n1][n2] += 1
            cooc_i[n2][n1] += 1
    cooc_flat_i = {}
    for n1 in cooc_i:
        for n2 in cooc_i[n1]:
            cooc_flat_i[(n1, n2)] = cooc_i[n1][n2]
    max_cooc_i = max(cooc_flat_i.values()) if cooc_flat_i else 1
    ultimo_sorteo_i = datos[-1]
    
    for n in range(46):
        delay = len(datos) - 1 - last_seen_i.get(n, -1)
        freq_5 = sum(1 for s_h in datos[-5:] if n in s_h) / 5.0
        freq_15 = sum(1 for s_h in datos[-15:] if n in s_h) / 15.0
        freq_30 = sum(1 for s_h in datos[-30:] if n in s_h) / 30.0
        cooc_last = sum(cooc_i[n].get(u, 0) for u in ultimo_sorteo_i) / (6.0 * max_cooc_i)
        X_test.append([delay, freq_5, freq_15, freq_30, cooc_last])
        
    probs = clf.predict_proba(X_test)[:, 1]
    
    ranking = [(n, probs[n]) for n in range(46)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    top_14 = sorted([x[0] for x in ranking[:14]])
    print(f"Top 14 Numeros ML: {top_14}")
    
    todas_combinaciones = list(itertools.combinations(top_14, 6))
    boletos_reducidos = [combo for combo in todas_combinaciones if es_ticket_perfecto(combo)]
    
    boletos_puntuados = []
    for boleto in boletos_reducidos:
        score_ind = sum(probs[n] for n in boleto)
        score_cooc = 0
        for n1, n2 in itertools.combinations(boleto, 2):
            score_cooc += cooc_i[n1].get(n2, 0) / max_cooc_i
            
        score_total = (score_ind * 0.7) + (score_cooc * 0.3)
        boletos_puntuados.append((boleto, score_total))
        
    boletos_puntuados.sort(key=lambda x: x[1], reverse=True)
    tickets_optimos = seleccionar_mmr(boletos_puntuados, 0.1)
    
    print("\n=== MODELO ML OPTIMO: 3 TICKETS (MMR p=0.10) ===")
    for idx, b in enumerate(tickets_optimos):
        print(f" Boleta #{idx+1} : [ " + " - ".join(f"{n:02d}" for n in sorted(b)) + " ]")
        
    # Guardar en historial_predicciones.json
    historial_path = 'c:/Users/Administrador/Desktop/lot/historial_predicciones.json'
    historial = []
    if os.path.exists(historial_path):
        with open(historial_path, 'r', encoding='utf-8') as f:
            historial = json.load(f)
            
    nueva_prediccion = {
        "fecha_prediccion": datetime.now().isoformat(),
        "fecha_sorteo_objetivo": "2026-05-27 (Miércoles)",
        "juego": "Quini 6 (Modelo ML Random Forest - 3 Tickets p=0.10)",
        "tickets": [
            {
                "nombre": f"SÚPER TICKET ML OPTIMO {chr(65+idx)}",
                "numeros": sorted(b)
            } for idx, b in enumerate(tickets_optimos)
        ]
    }
    
    historial.append(nueva_prediccion)
    with open(historial_path, 'w', encoding='utf-8') as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)
    print("\nPredicciones optimizadas guardadas exitosamente en historial_predicciones.json.")

if __name__ == '__main__':
    generar_proximos()
