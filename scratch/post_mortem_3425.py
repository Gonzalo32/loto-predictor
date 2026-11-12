import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from collections import Counter
from generador_cobertura_ml import load_data

draws = load_data()
print(f"Total draws loaded: {len(draws)}")
s3424 = [d for d in draws if d['sorteo'] == 3424]
s3425 = [d for d in draws if d['sorteo'] == 3425]

print("--- SORTEO 3424 ---")
for d in s3424:
    print(f"{d['modalidad']:<12}: {d['numbers']}")

print("\n--- SORTEO 3425 ---")
for d in s3425:
    print(f"{d['modalidad']:<12}: {d['numbers']}")

all_3424 = set(n for d in s3424 for n in d['numbers'])
all_3425 = set(n for d in s3425 for n in d['numbers'])

print(f"\nUnique numbers in 3424: {len(all_3424)}")
print(f"Unique numbers in 3425: {len(all_3425)}")
common = all_3424 & all_3425
print(f"Common numbers between date 3424 and date 3425: {sorted(list(common))} (Total: {len(common)})")

counts_3424 = Counter(n for d in s3424 for n in d['numbers'])
print("\nMulti-modal numbers in 3424:")
for n, c in counts_3424.most_common():
    if c >= 2:
        print(f"  Ball {n:02d}: {c} modalities (In 3425: {n in all_3425})")

counts_3425 = Counter(n for d in s3425 for n in d['numbers'])
print("\nMulti-modal numbers in 3425:")
for n, c in counts_3425.most_common():
    if c >= 2:
        print(f"  Ball {n:02d}: {c} modalities (In 3424: {n in all_3424})")

# Check decade distribution in 3425
decades_3425 = Counter(n // 10 for d in s3425 for n in d['numbers'])
print("\nDecade distribution across all 4 modalities in 3425 (Total balls = 24):")
for dec in range(5):
    range_str = f"{dec*10:02d}-{min(dec*10+9, 45):02d}"
    print(f"  Decade {dec} ({range_str}): {decades_3425[dec]} balls ({decades_3425[dec]/24*100:.1f}%)")

# Check user tickets
tickets_jugados = [
    [1, 7, 12, 26, 27, 31],
    [1, 7, 14, 25, 41, 43],
    [5, 7, 14, 21, 31, 39]
]
print("\nAudit of user tickets on 3425:")
for idx, t in enumerate(tickets_jugados, 1):
    hits_per_mod = {d['modalidad']: set(t) & set(d['numbers']) for d in s3425}
    pe = set(t) & (set(s3425[0]['numbers']) | set(s3425[1]['numbers']) | set(s3425[2]['numbers']))
    print(f"Boleto {idx} {t}:")
    for mod, h in hits_per_mod.items():
        print(f"  {mod:<12}: {len(h)} aciertos {sorted(list(h)) if h else ''}")
    print(f"  Pozo Extra  : {len(pe)} aciertos {sorted(list(pe)) if pe else ''}")
