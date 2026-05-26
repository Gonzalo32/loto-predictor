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

def generar_proximos():
    datos = leer_datos()
    if not datos:
        print("Error al leer datos.")
        return
        
    print(f"Analizando {len(datos)} sorteos historicos...")
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
    
    print(f"Top 13 Numeros: {top_13}")
    
    todas_combinaciones = list(itertools.combinations(top_13, 6))
    boletos_reducidos = [combo for combo in todas_combinaciones if es_ticket_perfecto(combo)]
    
    boletos_puntuados = []
    for boleto in boletos_reducidos:
        score_ind = sum(puntaje_final[n] for n in boleto)
        score_cooc = 0
        for n1, n2 in itertools.combinations(boleto, 2):
            score_cooc += cooc[n1].get(n2, 0) / max_cooc
        
        score_total = (score_ind * 0.7) + (score_cooc * 0.3)
        boletos_puntuados.append((boleto, score_total))
        
    boletos_puntuados.sort(key=lambda x: x[1], reverse=True)
    mejores_15 = [list(x[0]) for x in boletos_puntuados[:15]]
    
    print("\n=== TOP 15 TICKETS POR AFINIDAD Y SCORE ===")
    for idx, boleto in enumerate(mejores_15):
        boleto_str = " - ".join(f"{n:02d}" for n in sorted(boleto))
        score = boletos_puntuados[idx][1]
        print(f" Boleta #{idx+1:02d} : [ {boleto_str} ]  (Score: {score:.4f})")
        
    # Guardar en historial_predicciones.json
    historial_path = 'c:/Users/Administrador/Desktop/lot/historial_predicciones.json'
    historial = []
    if os.path.exists(historial_path):
        with open(historial_path, 'r', encoding='utf-8') as f:
            historial = json.load(f)
            
    nueva_prediccion = {
        "fecha_prediccion": datetime.now().isoformat(),
        "fecha_sorteo_objetivo": "2026-05-27 (Miercoles)",
        "juego": "Quini 6 (Modelo FFT + Markov + Afinidad Apriori - 15 Tickets)",
        "tickets": [
            {
                "nombre": f"BOLETA AFINIDAD #{idx+1:02d}",
                "numeros": sorted(boleto)
            } for idx, boleto in enumerate(mejores_15)
        ]
    }
    historial.append(nueva_prediccion)
    with open(historial_path, 'w', encoding='utf-8') as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)
    print("\n✅ Predicciones de afinidad guardadas exitosamente en historial_predicciones.json")

if __name__ == '__main__':
    generar_proximos()
