import urllib.request
import re

url = "https://quini-6-resultados.com.ar/sorteo-3411/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8')
        print(f"Sorteo 3411 page length: {len(html)}")
        for line in html.split('\n'):
            if any(w in line.lower() for w in ['2026', '23', 'septiembre', 'setiembre', 'sorteo']):
                print("  ", line.strip()[:150])
except Exception as e:
    print("Error:", e)
