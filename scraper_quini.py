import urllib.request
import re
import os
import sys
from typing import TypedDict, List, Dict, Optional

class SorteoData(TypedDict):
    sorteo: int
    fecha: str
    Tradicional: List[str]
    Segunda: List[str]
    Revancha: List[str]
    SiempreSale: List[str]

BASE_URL = "https://www.quini-6-resultados.com.ar"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_LIMPIO_PATH = os.path.join(BASE_DIR, 'historico_quini_limpio.csv')
CSV_COMPLETO_PATH = os.path.join(BASE_DIR, 'historico_quini_completo.csv')

def fetch_html(url: str) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12) as response:
            return response.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"[ERROR] No se pudo acceder a {url}: {e}")
        return None

def get_last_sorteo_in_csv() -> int:
    if not os.path.exists(CSV_LIMPIO_PATH):
        return 0
    with open(CSV_LIMPIO_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if parts and parts[0].isdigit():
                return int(parts[0])
    return 0

def parse_sorteo_html(html: str, sorteo_num: int, fecha_str: str) -> Optional[SorteoData]:
    modalidades: Dict[str, List[str]] = {}
    pattern1 = r'<span class="sorteo">\s*([^<]+)\s*</span>.*?<td class="nro">\s*([\d\s\-]+)\s*</td>'
    matches1 = re.findall(pattern1, html, re.DOTALL | re.IGNORECASE)
    
    pattern2 = r'<h3[^>]*>\s*([^<]+)\s*</h3>\s*<p class="numeros">\s*([\d\s\-]+)\s*</p>'
    matches2 = re.findall(pattern2, html, re.DOTALL | re.IGNORECASE)
    
    for mod_raw, nums_raw in matches1 + matches2:
        mod_clean = mod_raw.strip().upper()
        nums = [n.strip().zfill(2) for n in nums_raw.split('-') if n.strip().isdigit()]
        if len(nums) == 6:
            if 'TRADICIONAL' in mod_clean and 'SEGUNDA' not in mod_clean:
                modalidades['Tradicional'] = nums
            elif 'SEGUNDA' in mod_clean:
                modalidades['Segunda'] = nums
            elif 'REVANCHA' in mod_clean:
                modalidades['Revancha'] = nums
            elif 'SIEMPRE' in mod_clean:
                modalidades['SiempreSale'] = nums

    req_keys = ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']
    if all(k in modalidades for k in req_keys):
        return {
            'sorteo': sorteo_num,
            'fecha': fecha_str,
            'Tradicional': modalidades['Tradicional'],
            'Segunda': modalidades['Segunda'],
            'Revancha': modalidades['Revancha'],
            'SiempreSale': modalidades['SiempreSale']
        }
    return None

def fetch_pending_sorteos() -> List[SorteoData]:
    last_csv_sorteo = get_last_sorteo_in_csv()
    print(f"[INFO] Último sorteo registrado en CSV local: #{last_csv_sorteo}")
    
    main_html = fetch_html(BASE_URL + "/")
    if not main_html:
        return []
    
    pattern = r'href=["\'](?:https?://[^/]+)?(/quini6/sorteo-(\d+)-del-dia-(\d{2})-(\d{2})-(\d{4})\.htm)["\']'
    matches = re.findall(pattern, main_html, re.IGNORECASE)
    
    pending_items = []
    seen = set()
    for rel_link, s_num_str, dd, mm, yyyy in matches:
        s_num = int(s_num_str)
        if s_num > last_csv_sorteo and s_num not in seen:
            seen.add(s_num)
            fecha_iso = f"{yyyy}-{mm}-{dd}"
            pending_items.append((s_num, fecha_iso, rel_link))
            
    # Sort descending by sorteo number
    pending_items.sort(key=lambda x: x[0], reverse=True)
    
    if not pending_items:
        print("[OK] Los CSVs históricos ya están 100% actualizados con los últimos sorteos.")
        return []
        
    print(f"[INFO] Se encontraron {len(pending_items)} sorteos nuevos pendientes de actualización.")
    new_data: List[SorteoData] = []
    
    for s_num, fecha_iso, rel_link in pending_items:
        print(f" -> Descargando Sorteo #{s_num} ({fecha_iso})...")
        s_html = fetch_html(BASE_URL + rel_link)
        if s_html:
            s_data = parse_sorteo_html(s_html, s_num, fecha_iso)
            if s_data:
                new_data.append(s_data)
                print(f"    [ÉXITO] Sorteo #{s_num} procesado correctamente.")
            else:
                print(f"    [WARN] No se pudieron extraer las 4 modalidades para Sorteo #{s_num}.")
                
    return new_data

def actualizar_csv_completo(nuevos: List[SorteoData]):
    if not nuevos:
        return
    with open(CSV_COMPLETO_PATH, 'r', encoding='utf-8') as f:
        existing_lines = f.readlines()

    header = existing_lines[0]
    min_new_sorteo = min(s['sorteo'] for s in nuevos)
    
    filtered_lines = [
        line for line in existing_lines[1:]
        if not (line.strip().split(',')[0].isdigit() and int(line.strip().split(',')[0]) >= min_new_sorteo)
    ]

    new_rows = []
    for s in nuevos:
        for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
            row_str = f"{s['sorteo']},{s['fecha']},{mod}," + ",".join(s[mod]) + "\n"
            new_rows.append(row_str)

    with open(CSV_COMPLETO_PATH, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(new_rows)
        f.writelines(filtered_lines)
    print(f"[OK] {CSV_COMPLETO_PATH} actualizado exitosamente.")

def actualizar_csv_limpio(nuevos: List[SorteoData]):
    if not nuevos:
        return
    with open(CSV_LIMPIO_PATH, 'r', encoding='utf-8') as f:
        existing_lines = f.readlines()

    header = existing_lines[0]
    min_new_sorteo = min(s['sorteo'] for s in nuevos)

    filtered_lines = [
        line for line in existing_lines[1:]
        if not (line.strip().split(',')[0].isdigit() and int(line.strip().split(',')[0]) >= min_new_sorteo)
    ]

    new_rows = []
    for s in nuevos:
        row_str = f"{s['sorteo']},{s['fecha']}," + ",".join(s['Tradicional']) + ",00\n"
        new_rows.append(row_str)

    with open(CSV_LIMPIO_PATH, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(new_rows)
        f.writelines(filtered_lines)
    print(f"[OK] {CSV_LIMPIO_PATH} actualizado exitosamente.")

def main():
    print("="*60)
    print(" SCRAPER AUTOMÁTICO DE RESULTADOS QUINI 6")
    print("="*60)
    nuevos = fetch_pending_sorteos()
    if nuevos:
        actualizar_csv_completo(nuevos)
        actualizar_csv_limpio(nuevos)
        print("\n[RESUMEN DE SORTEOS AGREGADOS]")
        for s in nuevos:
            print(f" - Sorteo #{s['sorteo']} ({s['fecha']}): Tradicional -> {s['Tradicional']}")
    print("="*60)

if __name__ == '__main__':
    main()
