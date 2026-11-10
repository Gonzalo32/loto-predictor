"""
=============================================================================
PIPELINE UNIFICADO AUTOMATIZADO - QUINI 6 (PRODUCCIÓN)
=============================================================================
Ejecuta el ciclo de vida completo en 1 solo comando:
1. Web Scraping: Consulta si hay nuevos sorteos en la web oficial.
2. Actualización Histórica: Si hay sorteo nuevo, actualiza historico_quini_completo.csv.
3. Auditoría Automática: Compara la predicción previa con los resultados reales.
4. Machine Learning & Optimización Combinatoria: Entrena el super-ensamble,
   aplica detección de régimen dinámico, macro-filtros gaussianos y guardián de solapamiento (Pool 10).
5. Generación del Cupón Oficial: 3 boletos especializados para el próximo sorteo.
6. Sincronización Automática: Actualiza proxima_prediccion.json, historial_auditorias.json
   y el Dashboard interactivo (cupon_quini6_dashboard.html).
7. Formato WhatsApp / Agencia: Emite el texto listo para enviar al quinielero.
=============================================================================
"""

import os
import sys
import json
import subprocess
from datetime import datetime

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from scraper_quini import fetch_pending_sorteos, actualizar_csv_completo, get_last_sorteo_in_csv
from generador_cobertura_ml import main as ejecutar_generador

def sincronizar_dashboard_html():
    """Actualiza automáticamente el artefacto HTML cupon_quini6_dashboard.html con la última predicción."""
    pred_path = os.path.join(WORKSPACE_DIR, 'proxima_prediccion.json')
    if not os.path.exists(pred_path):
        return
        
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred = json.load(f)
            
        tickets = pred.get('tickets', [])
        if len(tickets) < 3:
            return
            
        # Buscar archivo dashboard en brain o local
        brain_dirs = [
            r'C:\Users\Administrador\.gemini\antigravity\brain\23ec70d2-48aa-4ea5-9bc5-6cafd851815b\cupon_quini6_dashboard.html',
            os.path.join(WORKSPACE_DIR, 'cupon_quini6_dashboard.html')
        ]
        
        for dash_path in brain_dirs:
            if os.path.exists(dash_path):
                with open(dash_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Reemplazar tickets en el script JS
                t1 = tickets[0]['numeros']
                t2 = tickets[1]['numeros']
                t3 = tickets[2]['numeros']
                js_tickets_str = f"const tickets = [\n      {t1},\n      {t2},\n      {t3}\n    ];"
                
                import re
                content = re.sub(r'const tickets = \[\s*\[.*?\],\s*\[.*?\],\s*\[.*?\]\s*\];', js_tickets_str, content, flags=re.DOTALL)
                
                with open(dash_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"[OK] Dashboard sincronizado: {dash_path}")
    except Exception as e:
        print(f"[WARN] No se pudo sincronizar el dashboard HTML: {e}")

def emitir_formato_whatsapp():
    """Emite por consola el bloque listo para copiar y pegar a WhatsApp o al quinielero."""
    pred_path = os.path.join(WORKSPACE_DIR, 'proxima_prediccion.json')
    if not os.path.exists(pred_path):
        return
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred = json.load(f)
        
    tickets = pred.get('tickets', [])
    if len(tickets) < 3:
        return
        
    print("\n" + "="*60)
    print("📲 TEXTO LISTO PARA WHATSAPP / AGENCIA DE LOTERÍA")
    print("="*60)
    print("🎟️ QUINI 6 — CUPÓN OFICIAL INTELIGENTE")
    print(f"📅 Fecha objetivo: {pred.get('fecha_sorteo_objetivo', 'Próximo Sorteo')}")
    print("💰 Pozos Vacantes: Segunda ($2.464M) | Revancha ($1.691M)")
    print("-" * 60)
    for idx, t in enumerate(tickets, 1):
        nums_str = " - ".join(f"{n:02d}" for n in t['numeros'])
        print(f"👉 Boleto {idx}: {nums_str}")
    print("-" * 60)
    print("📌 Modalidad: Tradicional + Segunda + Revancha + Siempre Sale")
    print("💵 Costo cupón oficial: 1 Slip (3 jugadas completas = $1.500 ARS)")
    print("="*60 + "\n")

def run_pipeline():
    print("==================================================")
    print("🚀 INICIANDO PIPELINE UNIFICADO DE QUINI 6")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================================")
    
    # 1. Scraping & Actualización
    print("\n[PASO 1/4] Verificando nuevos sorteos en la web oficial...")
    last_csv_sorteo = get_last_sorteo_in_csv()
    print(f" -> Último sorteo en base de datos local: #{last_csv_sorteo}")
    
    try:
        nuevos = fetch_pending_sorteos()
        if nuevos:
            actualizar_csv_completo(nuevos)
            print(f" -> [NOVEDAD] ¡Se descargaron y guardaron {len(nuevos)} sorteos nuevos!")
        else:
            print(" -> Base de datos al día con la web oficial.")
    except Exception as e:
        print(f" -> [AVISO] Scraping no disponible o sin conexión ({e}). Continuando con base local.")
        
    new_last_sorteo = get_last_sorteo_in_csv()
        
    # 2. Ejecutar Generador ML & Auditoría
    print("\n[PASO 2/4] Ejecutando Ensamble de IA, Auditoría y Guardián de Solapamiento...")
    ejecutar_generador(target_mod='Tradicional')
    
    # 3. Sincronizar Dashboard
    print("\n[PASO 3/4] Sincronizando visualización interactiva...")
    sincronizar_dashboard_html()
    
    # 4. Formato WhatsApp
    print("\n[PASO 4/4] Generando ficha de agencia para el usuario...")
    emitir_formato_whatsapp()
    
    print("✅ PIPELINE FINALIZADO CON ÉXITO.")

if __name__ == '__main__':
    run_pipeline()
