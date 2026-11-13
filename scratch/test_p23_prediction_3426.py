import os, sys, itertools, math
from collections import Counter
import numpy as np
from datetime import datetime

WORKSPACE_DIR = r'c:\Users\Administrador\Desktop\lot'
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    es_ticket_valido, compute_antagonist_pairs
)
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
target_s = 3426
target_weekday = 6 # Domingo

train_draws = [d for d in draws if d['sorteo'] < target_s]
train_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional']

p_attn = compute_fast_attention_mlp_scores(train_trad)
counts_t = Counter(n for d in train_trad[-60:] for n in d['numbers'])
p_t = np.array([counts_t[n] for n in range(46)], dtype=float)
p_t /= (p_t.sum() + 1e-6)

c_cross = Counter(n for d in train_draws[-60:] for n in d['numbers'])
p_cross = np.array([c_cross[n] for n in range(46)], dtype=float)
p_cross /= (p_cross.sum() + 1e-6)

counts_pcr = Counter(n for d in train_trad[-20:] for n in d['numbers'])
p_pcr = np.array([counts_pcr[n] for n in range(46)], dtype=float)
p_pcr /= (p_pcr.sum() + 1e-6)

ddm = compute_delta_delay_momentum_scores(train_trad)
p_ddm = np.array([ddm[n] for n in range(46)], dtype=float)
p_ddm /= (p_ddm.sum() + 1e-6)

recent_trad = train_trad[-10:]
reps = [len(set(recent_trad[k]['numbers']) & set(recent_trad[k-1]['numbers'])) for k in range(1, len(recent_trad))]
avg_rep = np.mean(reps) if reps else 0.8
w_rec = 2.8 if avg_rep >= 1.0 else (1.2 if avg_rep <= 0.5 else 2.0)
w_ddm = 1.2 if avg_rep >= 1.0 else (2.8 if avg_rep <= 0.5 else 1.8)

borda_pure = {}
for n in range(46): borda_pure[n] = 0.0
models = [(p_attn, 3.0), (p_t, w_rec), (p_cross, 1.5), (p_pcr, 1.4), (p_ddm, w_ddm)]
for p_mod, w in models:
    rk = sorted(range(46), key=lambda x: p_mod[x], reverse=True)
    for rank, n in enumerate(rk):
        borda_pure[n] += w * (45 - rank)
        
last_s = max(d['sorteo'] for d in train_draws)
last_date_draws = [d for d in train_draws if d['sorteo'] == last_s]
sat_decades = set()
for d in last_date_draws:
    if d['modalidad'] == 'Tradicional':
        d_counts = Counter(num // 10 for num in d['numbers'])
        for dec, cnt in d_counts.items():
            if cnt >= 3: sat_decades.add(dec)

same_day_draws = [d for d in train_trad if datetime.strptime(d['fecha'], '%Y-%m-%d').weekday() == target_weekday]
counts_day = Counter(n for d in same_day_draws[-35:] for n in d['numbers'])

_, cooccur_mat = compute_exponential_cooccurrence_scores(train_trad, window=40, half_life=15.0)

borda_v = borda_pure.copy()
if sat_decades:
    for n in range(46):
        if n // 10 in sat_decades: borda_v[n] -= 50.0

ranked = sorted(range(46), key=lambda x: borda_v[x], reverse=True)
t1, t2, t3, t4 = ranked[0:9], ranked[9:18], ranked[18:27], ranked[27:36]

dens_circ = np.zeros(46)
for d in train_trad[-15:]:
    for n in d['numbers']: dens_circ[n] += 1.0
kernel_sech = np.array([1.0 / (math.cosh(dx/1.5)**2) for dx in range(-4, 5)])
kernel_sech /= kernel_sech.sum()
soliton_field = np.zeros(46)
for n in range(46):
    s_val = sum(dens_circ[(n + dx) % 46] * kernel_sech[k_idx] for k_idx, dx in enumerate(range(-4, 5)))
    soliton_field[n] = s_val
borda_sol = borda_v.copy()
for n in range(46): borda_sol[n] += 10.0 * soliton_field[n]
rk_sol = sorted(range(46), key=lambda x: borda_sol[x], reverse=True)
t1_sol, t2_sol, t3_sol, t4_sol = rk_sol[0:9], rk_sol[9:18], rk_sol[18:27], rk_sol[27:36]

triplet_counts = Counter()
for d in train_trad[-40:]:
    for tr in itertools.combinations(sorted(d['numbers']), 3):
        triplet_counts[tr] += 1

p_renyi = (p_t + 1e-5) / (p_t.sum() + 46*1e-5)
renyi_entropy = -math.log(np.sum(p_renyi**2) + 1e-6)
target_ov = 4 if renyi_entropy < 3.55 else 3

E0_res = 138.0
gamma_half = 15.0

double_hits = Counter()
for s_idx in range(target_s - 30, target_s):
    s_draws = [d for d in train_draws if d['sorteo'] == s_idx]
    if len(s_draws) >= 2:
        all_s_nums = Counter(n for d in s_draws for n in d['numbers'])
        for n, cnt in all_s_nums.items():
            if cnt >= 2: double_hits[n] += 1

F47_QR = set(pow(x, 2, 47) for x in range(1, 47))

# Sorteo previo para filtro de persistencia
prev_nums = train_trad[-1]['numbers']
prev_set = set(prev_nums)

antagonist_pairs = compute_antagonist_pairs(draws)
recent_numbers = set()
for d in draws[-5:]:
    recent_numbers.update(d['numbers'])

def get_cands_p23(t1_l, t2_l, t3_l, t4_l):
    cands = []
    for c_t1 in itertools.combinations(t1_l[:6], 3):
        for n2 in t2_l[:3]:
            for n3 in t3_l[:2]:
                for n4 in t4_l[:2]:
                    cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                    if not es_ticket_valido(cand, recent_numbers, sat_decades, antagonist_pairs, min_distinct_decades=3):
                        continue
                    qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                    if qr_c < 2 or qr_c > 4:
                        continue
                    # FILTRO DE PERSISTENCIA P2: Máximo 2 repetidos del sorteo anterior
                    if len(set(cand) & prev_set) > 2:
                        continue
                    cands.append(cand)
    if len(cands) < 3:
        cands = [c for c in itertools.combinations(t1_l[:6] + t2_l[:3], 6) if es_ticket_valido(c, min_distinct_decades=3)]
    return cands

def sc_p23(c, bd):
    s = sum(bd[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
    for tr in itertools.combinations(c, 3):
        s += 16.0 * triplet_counts.get(tr, 0)
    bw_amp = (gamma_half**2) / ((sum(c) - E0_res)**2 + gamma_half**2)
    s += 28.0 * bw_amp
    # MODULACIÓN DE PARIDAD MARKOVIANA P3 (bonus w=8.0 si tiene 2, 3 o 4 pares)
    n_ev = sum(1 for x in c if x % 2 == 0)
    if n_ev in (2, 3, 4):
        s += 8.0
    return s

cands_b1 = get_cands_p23(t1_sol, t2_sol, t3_sol, t4_sol)
b1 = max(cands_b1, key=lambda c: sc_p23(c, borda_sol))

cands_b2 = get_cands_p23(t1, t2, t3, t4)
borda_d = borda_v.copy()
for n in range(46): borda_d[n] += 6.0 * counts_day[n]
cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d))

p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
cands_b3 = get_cands_p23(t1, t2, t3, t4)
cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]

bd_b3 = borda_v.copy()
for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12))

print('='*70)
print('CUPÓN SORTEO #3426 BAJO MODELO P23 (PERSISTENCIA + PARIDAD MARKOV):')
print('='*70)
print(f'Rényi Entropy: {renyi_entropy:.4f} -> Target Overlap B1-B2: {target_ov}')
for idx, b in enumerate([b1, b2, b3], 1):
    ev = sum(1 for x in b if x % 2 == 0)
    rep_prev = len(set(b) & prev_set)
    print(f'Boleto {idx}: {[f"{n:02d}" for n in b]} | Suma: {sum(b)} | Pares: {ev} | Repite de #3425: {rep_prev} ({set(b) & prev_set})')

print(f'Solapamiento B1-B2: {len(set(b1) & set(b2))} nums ({sorted(list(set(b1) & set(b2)))})')
print(f'Solapamiento B1-B3: {len(set(b1) & set(b3))} nums ({sorted(list(set(b1) & set(b3)))})')
print(f'Solapamiento B2-B3: {len(set(b2) & set(b3))} nums ({sorted(list(set(b2) & set(b3)))})')
total_nums = set(b1) | set(b2) | set(b3)
print(f'Total números únicos cubiertos: {len(total_nums)} -> {sorted(list(total_nums))}')
print('='*70)

