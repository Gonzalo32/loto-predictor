import urllib.request
import re

url = "https://quini-6-resultados.com.ar/"
req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
)

try:
    with urllib.request.urlopen(req, timeout=10) as response:
        html = response.read().decode('utf-8')
        print(f"HTML Length: {len(html)}")
        
        # Search for Sorteo number
        sorteos = re.findall(r'Sorteo\s*N?°?\s*(\d{4})', html, re.IGNORECASE)
        print("Sorteos encontrados:", list(set(sorteos))[:5])
        
        # Print a snippet of HTML containing numbers or modalities
        matches = re.findall(r'Tradicional.*?(?=\n\s*\n|\Z)', html, re.DOTALL | re.IGNORECASE)
        if matches:
            print("\nSnippet Tradicional:\n", matches[0][:300])
except Exception as e:
    print("Error:", e)
