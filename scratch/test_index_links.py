import urllib.request
import re

url = "https://www.quini-6-resultados.com.ar/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode('utf-8')
    
    # Pattern for draw link: /quini6/sorteo-(\d+)-del-dia-(\d{2})-(\d{2})-(\d{4})\.htm
    pattern = r'href=["\'].*?sorteo-(\d+)-del-dia-(\d{2})-(\d{2})-(\d{4})\.htm["\']'
    matches = re.findall(pattern, html, re.IGNORECASE)
    
    print("Sorteos detectados en el index:")
    for num, dd, mm, yyyy in matches:
        print(f"  Sorteo #{num} | Fecha: {yyyy}-{mm}-{dd}")
