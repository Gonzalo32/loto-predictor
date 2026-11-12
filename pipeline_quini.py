"""
=============================================================================
PIPELINE UNIFICADO AUTOMATIZADO - QUINI 6 (PRODUCCION)
=============================================================================
Ejecuta el ciclo de vida completo en 1 solo comando:
1. Web Scraping: Consulta si hay nuevos sorteos en la web oficial.
2. Actualizacion Historica: Si hay sorteo nuevo, actualiza historico_quini_completo.csv.
3. Auditoria Automatica: Compara la prediccion previa con los resultados reales.
4. Machine Learning & Optimizacion Combinatoria: Entrena el super-ensamble,
   aplica deteccion de regimen dinamico, macro-filtros gaussianos y guardian de solapamiento (Pool 10).
5. Generacion del Cupon Oficial: 3 boletos especializados para el proximo sorteo.
6. Sincronizacion Automatica: Actualiza proxima_prediccion.json, historial_auditorias.json
   y el Dashboard interactivo (cupon_quini6_dashboard.html).
7. Formato WhatsApp / Agencia: Emite el texto listo para enviar al quinielero.
8. Historial Consolidado: Muestra el retorno acumulado de auditorias previas.
=============================================================================
"""

import os
import sys
import json
import re
from datetime import datetime

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from scraper_quini import fetch_pending_sorteos, actualizar_csv_completo, get_last_sorteo_in_csv
from generador_cobertura_ml import main as ejecutar_generador

def sincronizar_dashboard_html():
    """Actualiza automaticamente el artefacto HTML cupon_quini6_dashboard.html con la ultima prediccion."""
    pred_path = os.path.join(WORKSPACE_DIR, 'proxima_prediccion.json')
    if not os.path.exists(pred_path):
        return
        
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred = json.load(f)
            
        tickets = pred.get('tickets', [])
        if len(tickets) < 3:
            return
            
        brain_dirs = [
            r'C:\Users\Administrador\.gemini\antigravity\brain\23ec70d2-48aa-4ea5-9bc5-6cafd851815b\cupon_quini6_dashboard.html',
            os.path.join(WORKSPACE_DIR, 'cupon_quini6_dashboard.html')
        ]
        
        for dash_path in brain_dirs:
            if os.path.exists(dash_path):
                with open(dash_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                t1 = tickets[0]['numeros']
                t2 = tickets[1]['numeros']
                t3 = tickets[2]['numeros']
                js_tickets_str = f"const tickets = [\n      {t1},\n      {t2},\n      {t3}\n    ];"
                
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
    print("TEXTO LISTO PARA WHATSAPP / AGENCIA DE LOTERIA")
    print("="*60)
    print("QUINI 6 - CUPON OFICIAL INTELIGENTE")
    print(f"Fecha objetivo: {pred.get('fecha_sorteo_objetivo', 'Proximo Sorteo')}")
    print("Pozos Vacantes: Segunda ($2.464M) | Revancha ($1.691M)")
    print("-" * 60)
    for idx, t in enumerate(tickets, 1):
        nums_str = " - ".join(f"{n:02d}" for n in t['numeros'])
        print(f"Boleto {idx}: {nums_str}")
    print("-" * 60)
    print("Modalidad: Tradicional + Segunda + Revancha + Siempre Sale")
    print("Inversión: 3 cupones completos ($4.000 c/u = $12.000 ARS total)")
    print("="*60 + "\n")

def mostrar_resumen_auditorias():
    """Muestra un resumen consolidado de todos los sorteos auditados en historial_auditorias.json."""
    audit_path = os.path.join(WORKSPACE_DIR, 'historial_auditorias.json')
    if not os.path.exists(audit_path):
        print("[INFO] Aun no hay sorteos auditados registrados.")
        return
    try:
        with open(audit_path, 'r', encoding='utf-8') as f:
            audits = json.load(f)
        if not audits:
            print("[INFO] Historial de auditorias vacio.")
            return
            
        print("="*65)
        print("HISTORIAL CONSOLIDADO DE AUDITORIAS DE PRODUCCION")
        print("="*65)
        print(f"Total de sorteos auditados: {len(audits)}")
        for a in audits:
            s_num = a.get('sorteo')
            print(f"\nSorteo #{s_num}:")
            for t in a.get('tickets', []):
                t_nom = t.get('nombre', 'Boleto')
                nums = t.get('numeros', [])
                ac_trad = t.get('aciertos_tradicional', [])
                ac_seg  = t.get('aciertos_segunda', [])
                ac_ss   = t.get('aciertos_siempre_sale', [])
                ac_pe   = t.get('aciertos_pozo_extra', [])
                res_str = []
                if len(ac_trad) >= 3: res_str.append(f"Tradicional: {len(ac_trad)} {ac_trad} [PREMIO]")
                if len(ac_seg) >= 3: res_str.append(f"Segunda: {len(ac_seg)} {ac_seg} [PREMIO]")
                if len(ac_ss) >= 3: res_str.append(f"Siempre Sale: {len(ac_ss)} {ac_ss} [PREMIO]")
                if len(ac_pe) >= 3: res_str.append(f"Pozo Extra: {len(ac_pe)} {ac_pe}")
                summary = " | ".join(res_str) if res_str else f"Trad: {len(ac_trad)} | SS: {len(ac_ss)}"
                print(f"  * {t_nom[:35]:<35} -> {summary}")
        print("="*65 + "\n")
    except Exception as e:
        print(f"[WARN] Error al leer historial de auditorias: {e}")

def run_pipeline(target_mod='Tradicional'):
    print("==================================================")
    print("INICIANDO PIPELINE UNIFICADO DE QUINI 6")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modalidad Objetivo: {target_mod}")
    print("==================================================")
    
    # 1. Scraping & Actualizacion
    print("\n[PASO 1/4] Verificando nuevos sorteos en la web oficial...")
    last_csv_sorteo = get_last_sorteo_in_csv()
    print(f" -> Ultimo sorteo en base de datos local: #{last_csv_sorteo}")
    
    try:
        nuevos = fetch_pending_sorteos()
        if nuevos:
            actualizar_csv_completo(nuevos)
            print(f" -> [NOVEDAD] Se descargaron y guardaron {len(nuevos)} sorteos nuevos!")
        else:
            print(" -> Base de datos al dia con la web oficial.")
    except Exception as e:
        print(f" -> [AVISO] Scraping no disponible o sin conexion ({e}). Continuando con base local.")
        
    new_last_sorteo = get_last_sorteo_in_csv()
        
    # 2. Ejecutar Generador ML & Auditoria
    print("\n[PASO 2/4] Ejecutando Ensamble de IA, Auditoria y Guardian de Solapamiento...")
    ejecutar_generador(target_mod=target_mod)
    
    # 3. Sincronizar Dashboard
    print("\n[PASO 3/4] Sincronizando visualizacion interactiva...")
    sincronizar_dashboard_html()
    
    # 4. Formato WhatsApp
    print("\n[PASO 4/4] Generando ficha de agencia para el usuario...")
    emitir_formato_whatsapp()
    
    # Resumen de Auditorias
    mostrar_resumen_auditorias()
    
    print("[OK] PIPELINE FINALIZADO CON EXITO.")

if __name__ == '__main__':
    target = 'Tradicional'
    if '--segunda' in sys.argv:
        target = 'Segunda'
    elif '--revancha' in sys.argv:
        target = 'Revancha'
    elif '--siempre-sale' in sys.argv:
        target = 'SiempreSale'
    run_pipeline(target_mod=target)
