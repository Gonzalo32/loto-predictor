from scraper import LotoScraper
from predictor import LotoPredictor
import time

def imprimir_ticket(titulo, numeros, plus):
    print("--------------------------------------------------")
    print(f"🌟 {titulo}")
    print("--------------------------------------------------")
    print(f"🎱 BOLIILAS: [ " + "  ".join(f"{n:02d}" for n in numeros) + " ]")
    print(f"⭐ PLUS:     [ {plus:02d} ]")
    print("--------------------------------------------------\n")

def main():
    print("==================================================")
    print("🎰 PREDICTOR AUTOMÁTICO - LOTO PLUS ARGENTINA 🇦🇷")
    print("==================================================")
    
    # 1. Recolectar datos
    scraper = LotoScraper()
    datos = scraper.collect_data()
    
    if not datos:
        print("❌ Error: No hay suficientes datos para iniciar las predicciones.")
        return

    # Añadimos un dato simulado recien extraido (como si fuera web scraping en vivo)
    print("🔄 Simulando actualización web de la jugada del Loto Plus...")
    time.sleep(1.5)
    ultimo_sorteo = int(datos[-1]['Sorteo'])
    nuevo = scraper.mock_fetch_new_draw(ultimo_sorteo)
    datos.append(nuevo)
    print(f"✅ Nuevo sorteo detectado e incorporado! Sorteo: #{nuevo['Sorteo']}\n")

    # 2. Inicializar Predictor
    print("🧠 Procesando Inteligencia de Datos y calculando patrones...")
    time.sleep(1)
    
    predictor = LotoPredictor(datos)

    # 3. Mostrar Sugerencias
    calientes, plus_caliente = predictor.sugerir_calientes()
    frios, plus_frio = predictor.sugerir_frios()
    mixto, plus_mixto = predictor.sugerir_mixto_balanceado()

    print("\n🔮 ESTADÍSTICA DE PREDICCIÓN CONCLUIDA:")
    imprimir_ticket("NÚMEROS CALIENTES (Mayores Apariciones)", calientes, plus_caliente)
    imprimir_ticket("NÚMEROS FRÍOS (Menores Apariciones, Ley de Retardo)", frios, plus_frio)
    imprimir_ticket("JUGADA EQUILIBRADA (Ratio Optimo)", mixto, plus_mixto)

    print("⚠️ Recuerda: El Loto Plus es un juego de azar matemáticamente independiente.")
    print("El uso de este software es estadístico y lúdico. ¡No juegues más de lo que puedas!")
    
if __name__ == "__main__":
    main()
