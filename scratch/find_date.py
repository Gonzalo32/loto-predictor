import urllib.request
import re

url = "https://quini-6-resultados.com.ar/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode('utf-8')
    print("Fecha snippets:")
    for line in html.split('\n'):
        if any(w in line.lower() for w in ['fecha', 'miércoles', 'miercoles', 'domingo', 'septiembre', 'setiembre', '2026', '23/']):
            print("  ", line.strip()[:150])
