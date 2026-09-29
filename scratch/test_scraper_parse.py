import urllib.request
import re
from datetime import datetime

def parse_quini_page(html):
    # Extract Sorteo number
    sorteo_match = re.search(r'Sorteo\s*N?°?\s*(\d{4})', html, re.IGNORECASE)
    sorteo_num = int(sorteo_match.group(1)) if sorteo_match else None
    
    # Extract Fecha (e.g. 23 de Septiembre de 2026 or 23/09/2026)
    # Looking for date patterns in Spanish
    fecha_match = re.search(r'(\d{1,2})\s+de\s+([A-Za-z]+)\s+de\s+(\d{4})', html, re.IGNORECASE)
    
    months = {
        'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
        'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
        'septiembre': '09', 'setiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
    }
    
    fecha_str = None
    if fecha_match:
        day = fecha_match.group(1).zfill(2)
        month_name = fecha_match.group(2).lower()
        month = months.get(month_name, '01')
        year = fecha_match.group(3)
        fecha_str = f"{year}-{month}-{day}"
        
    # Extract modalities
    # Structure: <span class="sorteo">MODALIDAD</span> ... <td class="nro">05 - 07 - 18 - 25 - 30 - 41</td>
    modalidades = {}
    pattern = r'<span class="sorteo">\s*([^<]+)\s*</span>.*?<td class="nro">\s*([\d\s\-]+)\s*</td>'
    matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
    
    for mod_raw, nums_raw in matches:
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
                
    return {
        'sorteo': sorteo_num,
        'fecha': fecha_str,
        'modalidades': modalidades
    }

url = "https://quini-6-resultados.com.ar/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
with urllib.request.urlopen(req, timeout=10) as resp:
    data = parse_quini_page(resp.read().decode('utf-8'))
    print("Resultado parseado:", data)
