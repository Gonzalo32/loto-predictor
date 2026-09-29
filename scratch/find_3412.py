import urllib.request
import re

url = "https://www.quini-6-resultados.com.ar/quini6/sorteo-3412-del-dia-27-09-2026.htm"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')

# Search for all td/tr or numbers
print("Length of HTML:", len(html))
# find all occurrences of Tradicional, Segunda, Revancha, Siempre
for word in ['TRADICIONAL', 'SEGUNDA', 'REVANCHA', 'SIEMPRE']:
    pos = html.find(word)
    if pos != -1:
        print(f"\n--- {word} ---")
        print(html[pos:pos+250])
