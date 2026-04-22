from collections import Counter
import random

class LotoPredictor:
    def __init__(self, datos_historicos):
        """
        datos_historicos es una lista de diccionarios con llaves: B1, B2, B3, B4, B5, B6, Plus
        Loto plus tiene bolillas del 0 al 45.
        """
        self.datos = datos_historicos

    def analizar_frecuencias(self):
        todas_bolillas = []
        todos_plus = []
        
        for fila in self.datos:
            # Extraemos las 6 bolillas (ignoramos Sorteo y Fecha)
            bolillas = [
                int(fila['B1']), int(fila['B2']), int(fila['B3']), 
                int(fila['B4']), int(fila['B5']), int(fila['B6'])
            ]
            todas_bolillas.extend(bolillas)
            
            if 'Plus' in fila:
                todos_plus.append(int(fila['Plus']))

        # Conteo de ocurrencias
        contador_bolillas = Counter(todas_bolillas)
        contador_plus = Counter(todos_plus)

        return contador_bolillas, contador_plus

    def sugerir_calientes(self):
        """Sugerencia basada en los números que más han salido (Números Calientes)."""
        if not self.datos: return []
        contador_bolillas, contador_plus = self.calientes_frios_analisis()
        
        # Tomar los 6 números más frecuentes
        mas_comunes = [num for num, frec in contador_bolillas.most_common(6)]
        mas_comunes.sort()
        
        plus_comun = contador_plus.most_common(1)[0][0] if contador_plus else random.randint(0, 9)
        
        return mas_comunes, plus_comun

    def sugerir_frios(self):
        """Sugerencia basada en los números que menos han salido."""
        if not self.datos: return []
        contador_bolillas, contador_plus = self.calientes_frios_analisis()
        
        # Identificar las que menos han salido. Nos aseguramos de incluir las de freq 0 si no existen en el counter.
        frecuencias = dict(contador_bolillas)
        bolillas_todas = {i: frecuencias.get(i, 0) for i in range(46)}
        
        # Ordenar por menor frecuencia
        menos_comunes = sorted(bolillas_todas.items(), key=lambda x: x[1])
        seis_frias = [num for num, frec in menos_comunes[:6]]
        seis_frias.sort()
        
        plus_frio = sorted({i: dict(contador_plus).get(i, 0) for i in range(10)}.items(), key=lambda x:x[1])[0][0]
        
        return seis_frias, plus_frio

    def calientes_frios_analisis(self):
        return self.analizar_frecuencias()

    def sugerir_mixto_balanceado(self):
        """Sugerencia estadísticamente balanceada: 3 calientes, 2 fríos, 1 aleatorio."""
        if not self.datos: return []
        
        calientes, _ = self.sugerir_calientes()
        frios, _ = self.sugerir_frios()
        
        # Tomar mezcla cuidando de no repetir números
        seleccion = set()
        seleccion.update(random.sample(calientes, 3))
        
        frios_disponibles = [x for x in frios if x not in seleccion]
        if len(frios_disponibles) >= 2:
            seleccion.update(random.sample(frios_disponibles, 2))
        else:
            seleccion.update(frios_disponibles)
            
        # Rellenar con aleatorios hasta llegar a 6
        while len(seleccion) < 6:
            candidato = random.randint(0, 45)
            seleccion.add(candidato)
            
        lista_final = list(seleccion)
        lista_final.sort()
        
        # Mezclar plus aleatoriamente ponderado? Mejor un plus normal
        plus_random = random.randint(0, 9)
        
        return lista_final, plus_random
