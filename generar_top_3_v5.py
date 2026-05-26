import csv
import numpy as np
from collections import defaultdict
import itertools
import os
import json
from datetime import datetime

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

def analisis_fft(datos, max_num=45):
    n_sorteos = len(datos)
    scores = {}
    for num in range(max_num + 1):
        serie = np.zeros(n_sorteos)
        for i, sorteo in enumerate(datos):
            if num in sorteo:
                serie[i] = 1.0
        media = np.mean(serie)
        serie_centrada = serie - media
        fft_vals = np.fft.fft(serie_centrada)
        fft_freqs = np.fft.fftfreq(n_sorteos)
        mitad = n_sorteos // 2
        amplitudes = np.abs(fft_vals[:mitad])
        freqs = fft_freqs[:mitad]
        if len(amplitudes) > 0:
            amplitudes[0] = 0
            idx_max = np.argmax(amplitudes)
            freq_max = freqs[idx_max]
            amp_max = amplitudes[idx_max]
            fase = np.angle(fft_vals[idx_max])
            t_next = n_sorteos
            valor_esperado = amp_max * np.cos(2 * np.pi * freq_max * t_next + fase)
        else:
            valor_esperado = 0
        scores[num] = float(valor_esperado)
    return scores

def analisis_markov_2do_orden(datos, max_num=45):
    transiciones = defaultdict(lambda: defaultdict(int))
    for i in range(len(datos) - 1):
        sorteo_actual = datos[i]
        sorteo_siguiente = datos[i+1]
        for num_a in sorteo_actual:
            for num_b in sorteo_siguiente:
                transiciones[num_a][num_b] += 1
    ultimo_sorteo = datos[-1]
    scores_markov = {n: 0.0 for n in range(max_num + 1)}
    for num in ultimo_sorteo:
        total_t = sum(transiciones[num].values())
        if total_t > 0:
            for n_sig, count in transiciones[num].items():
                prob = count / total_t
                scores_markov[n_sig] += prob
    return scores_markov

def analizar_coocurrencia(datos, max_num=45):
    cooc = defaultdict(lambda: defaultdict(int))
    for sorteo in datos:
        for n1, n2 in itertools.combinations(sorteo, 2):
            cooc[n1][n2] += 1
            cooc[n2][n1] += 1
    return cooc

def min_max_norm(dic):
    valores = list(dic.values())
    if not valores: return dic
    min_v = min(valores)
    max_v = max(valores)
    rango = max_v - min_v if max_v != min_v else 1
    return {k: (v - min_v) / rango for k, v in dic.items()}

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
        
    scores_fft = analisis_fft(datos, 45)
    scores_markov = analisis_markov_2do_orden(datos, 45)
    cooc = analizar_coocurrencia(datos, 45)
    
    cooc_flat = {}
    for n1 in cooc:
        for n2 in cooc[n1]:
            cooc_flat[(n1, n2)] = cooc[n1][n2]
    max_cooc = max(cooc_flat.values()) if cooc_flat else 1
    
    norm_fft = min_max_norm(scores_fft)
    norm_markov = min_max_norm(scores_markov)
    
    puntaje_final = {}
    for n in range(46):
        puntaje = (norm_fft.get(n, 0) * 0.6) + (norm_markov.get(n, 0) * 0.4)
        puntaje_final[n] = puntaje
        
    ranking = sorted(puntaje_final.items(), key=lambda x: x[1], reverse=True)
    top_13 = sorted([x[0] for x in ranking[:13]])
    
    todas_combinaciones = list(itertools.combinations(top_13, 6))
    boletos_reducidos = [combo for combo in todas_combinaciones if es_ticket_perfecto(combo)]
    
    pares_espejo_clave = {(1, 10), (23, 32), (13, 31)}
    
    boletos_puntuados = []
    for boleto in boletos_reducidos:
        score_ind = sum(puntaje_final[n] for n in boleto)
        
        # 1. Coocurrencia
        score_cooc = 0
        for n1, n2 in itertools.combinations(boleto, 2):
            score_cooc += cooc[n1].get(n2, 0) / max_cooc
            
        # 2. Pares espejo
        score_espejo = 0
        for n1, n2 in itertools.combinations(boleto, 2):
            if (n1, n2) in pares_espejo_clave or (n2, n1) in pares_espejo_clave:
                score_espejo += 1.0
                
        # 3. Patrón de distancia (gaps de +9, +6, +7)
        score_gap = 0
        ticket_sort = sorted(boleto)
        for j in range(1, len(ticket_sort)):
            gap = ticket_sort[j] - ticket_sort[j-1]
            if gap in [9, 6, 7]:
                score_gap += 0.5
                
        score_total = (score_ind * 0.5) + (score_cooc * 0.2) + (score_espejo * 0.15) + (score_gap * 0.15)
        boletos_puntuados.append((boleto, score_total))
        
    boletos_puntuados.sort(key=lambda x: x[1], reverse=True)
    
    # 1. Enfoque A: Jackpot Puro (p=0.00)
    tickets_jackpot = seleccionar_mmr(boletos_puntuados, 0.0)
    
    # 2. Enfoque B: Balanceado (p=0.10)
    tickets_balanceados = seleccionar_mmr(boletos_puntuados, 0.1)
    
    print("\n=== ESTRATEGIA A: JACKPOT PURO (p=0.00) ===")
    for idx, b in enumerate(tickets_jackpot):
        print(f" Boleta #{idx+1} : [ " + " - ".join(f"{n:02d}" for n in sorted(b)) + " ]")
        
    print("\n=== ESTRATEGIA B: BALANCEADO JACKPOT/5-HITS (p=0.10) ===")
    for idx, b in enumerate(tickets_balanceados):
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
        "juego": "Quini 6 (Modelo V5 - 3 Tickets Jackpot p=0.00)",
        "tickets": [
            {
                "nombre": f"SÚPER TICKET JACKPOT {chr(65+idx)}",
                "numeros": sorted(b)
            } for idx, b in enumerate(tickets_jackpot)
        ]
    }
    
    nueva_prediccion_bal = {
        "fecha_prediccion": datetime.now().isoformat(),
        "fecha_sorteo_objetivo": "2026-05-27 (Miércoles)",
        "juego": "Quini 6 (Modelo V5 - 3 Tickets Balanceado p=0.10)",
        "tickets": [
            {
                "nombre": f"SÚPER TICKET BALANCEADO {chr(65+idx)}",
                "numeros": sorted(b)
            } for idx, b in enumerate(tickets_balanceados)
        ]
    }
    
    historial.append(nueva_prediccion)
    historial.append(nueva_prediccion_bal)
    
    with open(historial_path, 'w', encoding='utf-8') as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)
    print("\nPredicciones guardadas exitosamente.")

if __name__ == '__main__':
    generar_proximos()
