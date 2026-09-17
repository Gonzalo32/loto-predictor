import csv
import json
import os
import sys
import numpy as np

# Load clean dataset
path = 'c:/Users/Administrador/Desktop/lot/historico_quini_limpio.csv'
datos = []
with open(path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            b = [int(r[f'B{i}']) for i in range(1, 7)]
            datos.append({'sorteo': int(r['Sorteo']), 'fecha': r['Fecha'], 'numeros': b})
        except:
            pass

# datos is newest first
print(f"Total sorteos registrados en limpio: {len(datos)}")
print(f"Último sorteo registrado: #{datos[0]['sorteo']} ({datos[0]['fecha']}) -> {datos[0]['numeros']}")
print(f"Penúltimo sorteo registrado: #{datos[1]['sorteo']} ({datos[1]['fecha']}) -> {datos[1]['numeros']}")

