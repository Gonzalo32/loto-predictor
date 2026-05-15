import csv
import numpy as np
from collections import defaultdict
import itertools
import os
import sys

def leer_datos(archivo='historico_quini_limpio.csv'):
    datos = []
    if not os.path.exists(archivo):
        print(f"Error: No se encontró {archivo}")
        return []
        
    with open(archivo, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bolillas = [int(row[f'B{i}']) for i in range(1, 7)]
                datos.append(bolillas)
            except ValueError:
                continue
    datos.reverse() # Orden cronológico: índice 0 es el más antiguo
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

def min_max_norm(dic):
    valores = list(dic.values())
    if not valores: return dic
    min_v = min(valores)
    max_v = max(valores)
    rango = max_v - min_v if max_v != min_v else 1
    return {k: (v - min_v) / rango for k, v in dic.items()}

def generar_top_12(datos_historia):
    scores_fft = analisis_fft(datos_historia, 45)
    scores_markov = analisis_markov_2do_orden(datos_historia, 45)
    
    norm_fft = min_max_norm(scores_fft)
    norm_markov = min_max_norm(scores_markov)
    
    puntaje_final = {}
    for n in range(46):
        puntaje = (norm_fft.get(n, 0) * 0.6) + (norm_markov.get(n, 0) * 0.4)
        puntaje_final[n] = puntaje
        
    ranking = sorted(puntaje_final.items(), key=lambda x: x[1], reverse=True)
    return sorted([x[0] for x in ranking[:12]])

def es_ticket_perfecto(ticket):
    ticket = sorted(ticket)
    # 1. Sin consecutivos
    for i in range(1, len(ticket)):
        if ticket[i] - ticket[i-1] <= 1:
            return False
            
    # 2. Spread entre 20 y 42
    spread = ticket[5] - ticket[0]
    if spread < 20 or spread > 42:
        return False
        
    # 3. Decenas (max 2)
    decenas = defaultdict(int)
    for num in ticket:
        decenas[num // 10] += 1
    if any(count > 2 for count in decenas.values()):
        return False
        
    # 4. Paridad
    pares = sum(1 for n in ticket if n % 2 == 0)
    if pares == 0 or pares == 6:
        return False
        
    return True

def simular_backtest():
    datos_completos = leer_datos('historico_quini_limpio.csv')
    if not datos_completos:
        return
        
    inicio_idx = 100 # Empezamos a predecir a partir del sorteo 101
    total_sorteos = len(datos_completos)
    
    aciertos_totales = {4: 0, 5: 0, 6: 0}
    sorteos_jugados = 0
    
    print(f"Iniciando Backtesting desde el sorteo {inicio_idx} hasta {total_sorteos}...")
    
    for i in range(inicio_idx, total_sorteos):
        historia = datos_completos[:i]
        sorteo_real = set(datos_completos[i])
        
        top_12 = generar_top_12(historia)
        
        # Generar combinaciones y filtrar
        todas_combinaciones = list(itertools.combinations(top_12, 6))
        boletos_reducidos = [combo for combo in todas_combinaciones if es_ticket_perfecto(combo)]
        
        mejor_acierto = 0
        for boleto in boletos_reducidos:
            aciertos = len(set(boleto).intersection(sorteo_real))
            if aciertos > mejor_acierto:
                mejor_acierto = aciertos
                
        if mejor_acierto >= 4:
            aciertos_totales[mejor_acierto] = aciertos_totales.get(mejor_acierto, 0) + 1
            
        sorteos_jugados += 1
        
        # Progreso cada 50 sorteos
        if sorteos_jugados % 50 == 0:
            print(f"Progreso: {sorteos_jugados}/{total_sorteos - inicio_idx} simulados. "
                  f"Hits acumulados -> 4: {aciertos_totales.get(4,0)}, 5: {aciertos_totales.get(5,0)}, 6: {aciertos_totales.get(6,0)}")

    print("\n" + "="*50)
    print("🏆 RESULTADOS FINALES DEL BACKTESTING 🏆")
    print("="*50)
    print(f"Sorteos Simulados: {sorteos_jugados}")
    print(f"Promedio de boletos jugados por sorteo: ~{len(boletos_reducidos)} tickets (en el último paso).")
    print(f"✅ Veces con un boleto de 4 Aciertos: {aciertos_totales.get(4, 0)}")
    print(f"🔥 Veces con un boleto de 5 Aciertos: {aciertos_totales.get(5, 0)}")
    print(f"👑 Veces con un boleto de 6 Aciertos: {aciertos_totales.get(6, 0)}")
    print("="*50)

if __name__ == '__main__':
    simular_backtest()
