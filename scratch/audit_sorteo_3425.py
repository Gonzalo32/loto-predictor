target_trad = {2, 24, 25, 32, 37, 40}
target_seg  = {0, 6, 17, 28, 33, 37}
target_rev  = {15, 27, 36, 37, 40, 42}
target_ss   = {10, 27, 28, 33, 36, 38}
pozo_extra = target_trad | target_seg | target_rev

user_tickets = [
    ("Boleto 1", [1, 7, 12, 26, 27, 31]),
    ("Boleto 2", [1, 7, 14, 25, 41, 43]),
    ("Boleto 3", [5, 7, 14, 21, 31, 39])
]

print("=== AUDITORIA SORTEO #3425 (11/11/2026) ===")
print("Tradicional :", sorted(list(target_trad)))
print("Segunda     :", sorted(list(target_seg)))
print("Revancha    :", sorted(list(target_rev)))
print("Siempre Sale:", sorted(list(target_ss)))
print("Pozo Extra (18 nums):", sorted(list(pozo_extra)))
print()

for name, t in user_tickets:
    t_set = set(t)
    h_trad = t_set & target_trad
    h_seg = t_set & target_seg
    h_rev = t_set & target_rev
    h_ss = t_set & target_ss
    h_pe = t_set & pozo_extra
    print(f"{name}: {t}")
    print(f"  Tradicional : {len(h_trad)} aciertos -> {sorted(list(h_trad))}")
    print(f"  Segunda     : {len(h_seg)} aciertos -> {sorted(list(h_seg))}")
    print(f"  Revancha    : {len(h_rev)} aciertos -> {sorted(list(h_rev))}")
    print(f"  Siempre Sale: {len(h_ss)} aciertos -> {sorted(list(h_ss))}")
    print(f"  Pozo Extra  : {len(h_pe)} aciertos -> {sorted(list(h_pe))}")
    print()

# Check what the calibrated tickets had:
calib_tickets = [
    ("Calib B1", [1, 5, 21, 26, 27, 39]),
    ("Calib B2", [1, 5, 7, 14, 25, 43]),
    ("Calib B3", [5, 7, 14, 21, 31, 39])
]
print("--- COMPARACION CON BOLETOS CALIBRADOS ---")
for name, t in calib_tickets:
    t_set = set(t)
    h_trad = t_set & target_trad
    h_seg = t_set & target_seg
    h_rev = t_set & target_rev
    h_ss = t_set & target_ss
    h_pe = t_set & pozo_extra
    print(f"{name}: {t}")
    print(f"  Trad: {len(h_trad)} {sorted(list(h_trad))} | Seg: {len(h_seg)} | Rev: {len(h_rev)} {sorted(list(h_rev))} | SS: {len(h_ss)} {sorted(list(h_ss))} | PE: {len(h_pe)} {sorted(list(h_pe))}")

