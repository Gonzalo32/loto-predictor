import csv
import json
import numpy as np
from collections import defaultdict
import os

def leer_datos(archivo='historico_quini_limpio.csv'):
    datos = []
    if not os.path.exists(archivo):
        print(f"Error: No se encontró {archivo}")
        return []
        
    with open(archivo, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Quini 6 usa bolillas de 0 a 45
                bolillas = [int(row[f'B{i}']) for i in range(1, 7)]
                datos.append(bolillas)
            except ValueError:
                continue
    datos.reverse() # Orden cronológico: índice 0 es el más antiguo
    return datos

def analisis_fft(datos, max_num=45):
    """
    Calcula la Transformada de Fourier para cada número y obtiene un 'score'.
    Un score alto indica que el número está en fase para salir pronto.
    """
    n_sorteos = len(datos)
    scores = {}
    
    for num in range(max_num + 1):
        # Crear serie de tiempo binaria
        serie = np.zeros(n_sorteos)
        for i, sorteo in enumerate(datos):
            if num in sorteo:
                serie[i] = 1.0
                
        # Restar la media para centrar la serie
        media = np.mean(serie)
        serie_centrada = serie - media
        
        # Aplicar FFT
        fft_vals = np.fft.fft(serie_centrada)
        # Frecuencias correspondientes
        fft_freqs = np.fft.fftfreq(n_sorteos)
        
        # Tomar la mitad positiva (es simétrica)
        mitad = n_sorteos // 2
        amplitudes = np.abs(fft_vals[:mitad])
        freqs = fft_freqs[:mitad]
        
        # Ignorar la frecuencia 0 (ya está centrada)
        amplitudes[0] = 0
        
        # Buscar la frecuencia dominante
        idx_max = np.argmax(amplitudes)
        freq_max = freqs[idx_max]
        amp_max = amplitudes[idx_max]
        
        # Reconstruir la onda principal para ver el valor esperado en el *próximo* sorteo
        fase = np.angle(fft_vals[idx_max])
        t_next = n_sorteos
        valor_esperado = amp_max * np.cos(2 * np.pi * freq_max * t_next + fase)
        
        scores[num] = float(valor_esperado)
        
    return scores

def analisis_markov_2do_orden(datos, max_num=45):
    """
    Analiza qué números tienden a salir después de la aparición conjunta
    o condicional del sorteo inmediatamente anterior.
    """
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

def generar_matriz_avanzada():
    print("Iniciando Análisis Matemático Avanzado para Quini 6...")
    datos = leer_datos('historico_quini_limpio.csv')
    if not datos:
        return
        
    print(f"Sorteos analizados: {len(datos)}")
    
    scores_fft = analisis_fft(datos, 45)
    scores_markov = analisis_markov_2do_orden(datos, 45)
    
    def min_max_norm(dic):
        valores = list(dic.values())
        if not valores: return dic
        min_v = min(valores)
        max_v = max(valores)
        rango = max_v - min_v if max_v != min_v else 1
        return {k: (v - min_v) / rango for k, v in dic.items()}
        
    norm_fft = min_max_norm(scores_fft)
    norm_markov = min_max_norm(scores_markov)
    
    # Puntaje combinado: 60% FFT y 40% Markov
    puntaje_final = {}
    for n in range(46):
        puntaje = (norm_fft.get(n, 0) * 0.6) + (norm_markov.get(n, 0) * 0.4)
        puntaje_final[n] = puntaje
        
    ranking = sorted(puntaje_final.items(), key=lambda x: x[1], reverse=True)
    top_12 = [x[0] for x in ranking[:12]]
    
    resultado = {
        "dataset": "Quini 6",
        "total_sorteos_analizados": len(datos),
        "top_12_numeros": top_12,
        "ranking_completo": [{"numero": k, "score": round(v, 4)} for k, v in ranking]
    }
    
    with open('matriz_avanzada.json', 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=4)
        
    print(f"Análisis completado.\nLos 12 mejores números según FFT y Markov son: {top_12}")
    print("Se ha generado 'matriz_avanzada.json'.")

if __name__ == '__main__':
    generar_matriz_avanzada()
