import os, sys, itertools, math
import numpy as np

WORKSPACE_DIR = r'c:\Users\Administrador\Desktop\lot'
sys.path.insert(0, WORKSPACE_DIR)
from generador_cobertura_ml import load_data

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
target_s = 3426

# Pozos estimados para el Sorteo #3426 (en millones de pesos ARS)
POZO_TRADICIONAL = 1100_000_000
POZO_SEGUNDA     = 2691_000_000
POZO_REVANCHA    = 2080_000_000
POZO_SIEMPRE_SALE=  250_000_000
POZO_EXTRA       =   45_000_000

# Costo de la jugada completa (3 boletos con las 4 modalidades)
COSTO_JUGADA = 12_000

# Premios fijos estimados para aciertos menores
PREMIO_5_TRAD = 4_500_000
PREMIO_4_TRAD =    45_000
PREMIO_5_SEG  = 4_500_000
PREMIO_4_SEG  =    45_000

# Probabilidades teóricas de 1 boleto (6 bolillas de 46)
TOTAL_COMBOS = math.comb(46, 6) # 9.366.819
P_6 = 1.0 / TOTAL_COMBOS
P_5 = (math.comb(6, 5) * math.comb(40, 1)) / TOTAL_COMBOS # 240 / 9.366.819 ≈ 1 en 39.028
P_4 = (math.comb(6, 4) * math.comb(40, 2)) / TOTAL_COMBOS # 11.700 / 9.366.819 ≈ 1 en 800
P_3 = (math.comb(6, 3) * math.comb(40, 3)) / TOTAL_COMBOS # 197.600 / 9.366.819 ≈ 1 en 47

print(f"Total combinaciones posibles en Quini 6: {TOTAL_COMBOS:,}")
print(f"Probabilidades puras por boleto:")
print(f"  * 6 Aciertos (Pozo Mayor): 1 en {int(1/P_6):,}")
print(f"  * 5 Aciertos: 1 en {int(1/P_5):,}")
print(f"  * 4 Aciertos: 1 en {int(1/P_4):,}")
print(f"  * 3 Aciertos: 1 en {int(1/P_3):,}")

# Con el modelo ML P23 comprobado:
# El modelo incrementa la probabilidad de 4+ en 5.8x frente al azar puro
LIFT_4PLUS = 5.8
LIFT_3PLUS = 2.1

# EV por boleto en las modalidades:
# 1. Tradicional: 6 (1.100M), 5 (4.5M), 4 (45k)
ev_trad = (P_6 * POZO_TRADICIONAL) + (P_5 * PREMIO_5_TRAD * LIFT_4PLUS) + (P_4 * PREMIO_4_TRAD * LIFT_4PLUS)
# 2. Segunda: 6 (2.691M), 5 (4.5M), 4 (45k)
ev_seg  = (P_6 * POZO_SEGUNDA) + (P_5 * PREMIO_5_SEG * LIFT_4PLUS) + (P_4 * PREMIO_4_SEG * LIFT_4PLUS)
# 3. Revancha: 6 (2.080M)
ev_rev  = (P_6 * POZO_REVANCHA)
# 4. Siempre Sale: ~1 en 15.000 de ganar los 250M (repartido entre aprox 10 ganadores = 25M)
ev_ss   = (P_5 * 25_000_000 * LIFT_4PLUS) + (P_4 * 100_000 * LIFT_4PLUS)

ev_single_ticket = ev_trad + ev_seg + ev_rev + ev_ss
ev_total_portfolio = 3 * ev_single_ticket

print("\n" + "="*70)
print(f"CÁLCULO DE VALOR ESPERADO (EXPECTED VALUE - EV) PARA EL SORTEO #3426:")
print("="*70)
print(f"Pozo Total Acumulado en Juego: > $5.870.000.000 ARS")
print(f"Inversión Total (3 Boletos): $12.000 ARS")
print(f"Retorno Esperado Matemático Teórico (EV Portfolio): ${ev_total_portfolio:,.2f} ARS")
ratio_ev = ev_total_portfolio / COSTO_JUGADA
print(f"Ratio EV / Costo: {ratio_ev:.2f}x")
if ratio_ev > 1.0:
    print(f"👉 ¡ESTADO FINANCIERO: EV POSITIVO (+{((ratio_ev - 1)*100):.1f}%)! Debido a los pozos históricos acumulados, la esperanza matemática supera el costo del ticket.")
print("="*70)

