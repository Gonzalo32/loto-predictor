import urllib.request
import re

url = "https://quini-6-resultados.com.ar/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode('utf-8')
    hrefs = re.findall(r'href=["\'](.*?)["\']', html, re.IGNORECASE)
    print("Hrefs found:")
    for h in hrefs[:30]:
        print("  ", h)
