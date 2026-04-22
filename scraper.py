import csv
import os
import random
from datetime import datetime, timedelta

class LotoScraper:
    def __init__(self, filename="historico_loto.csv"):
        self.filename = filename
        
    def collect_data(self):
        """
        Lee los datos historicos existentes. 
        En un escenario real iteraría librerias web (como BeautifulSoup y requests)
        para extraer sorteos nuevos de sitios como TuJugada.com.ar o la web oficial
        y los guardaría en el CSV antes de retornarlos.
        """
        print("🔎 [Scraper] Verificando fuente de datos y sorteos...")
        
        datos = []
        if os.path.exists(self.filename):
            with open(self.filename, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    datos.append(row)
            print(f"✅ [Scraper] {len(datos)} sorteos cargados desde el historial local.")
        else:
            print("⚠️ [Scraper] No se encontró historial. Se recomienda recopilar datos inicialmente.")
        
        return datos

    def mock_fetch_new_draw(self, last_draw_num):
        """
        Función para simular la llegada de un nuevo sorteo.
        """
        numeros = random.sample(range(0, 46), 6)
        numeros.sort()
        plus = random.randint(0, 9)
        nuevo_sorteo = {
            'Sorteo': str(last_draw_num + 1),
            'Fecha': (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
            'B1': f"{numeros[0]:02d}", 'B2': f"{numeros[1]:02d}", 'B3': f"{numeros[2]:02d}",
            'B4': f"{numeros[3]:02d}", 'B5': f"{numeros[4]:02d}", 'B6': f"{numeros[5]:02d}",
            'Plus': f"{plus:02d}"
        }
        return nuevo_sorteo
