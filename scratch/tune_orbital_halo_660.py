import os, sys, itertools, time, math
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import (
    load_data, compute_antagonist_pairs,
    compute_exponential_cooccurrence_scores, compute_delta_delay_momentum_scores,
    es_ticket_valido
)
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
n_eval = 660
eval_draws = trad_draws[-n_eval:]

F47_QR = set(pow(x, 2, 47) for x in range(1, 47))

# Calibración Fina de la Teoría del Halo Orbital
configs = {
    'H1_ref (12, 10)':                 {'b3_w1': 12.0, 'b3_w2': 10.0, 'b3_w3':  0.0, 'b2_halo': False},
    'H1_v1 (14, 12)':                  {'b3_w1': 14.0, 'b3_w2': 12.0, 'b3_w3':  0.0, 'b2_halo': False},
    'H1_v2 (10, 8)':                   {'b3_w1': 10.0, 'b3_w2':  8.0, 'b3_w3':  0.0, 'b2_halo': False},
    'H1_v3 (14, 11, 6)':               {'b3_w1': 14.0, 'b3_w2': 11.0, 'b3_w3':  6.0, 'b2_halo': False},
    'H1_v4 (12, 10 + B2 soft halo)':   {'b3_w1': 12.0, 'b3_w2': 10.0, 'b3_w3':  0.0, 'b2_halo': True},
    'H1_v5 (16, 14)':                  {'b3_w1': 16.0, 'b3_w2': 14.0, 'b3_w3':  0.0, 'b2_halo': False}
}

results = {c: {'trad_4': 0, 'glob_4': 0, 'trad_3': 0, 'glob_3': 0, 'pe_5': 0, 'secos': 0} for c in configs}

print("="*115)
print(f"CALIBRACIÓN FINA DEL HALO ORBITAL DE LIE ({n_eval} SORTEOS / 1.980 BOLETOS)")
print("Optimizando pesos de perturbación local para maximizar 4+...")
print("="*115)

start_time = time.time()

for idx_eval, target_draw in enumerate(eval_draws):
    target_s = target_draw['sorteo']
    target_date_str = target_draw['fecha']
    target_weekday = datetime.strptime(target_date_str, '%Y-%m-%d').weekday()
    
    train_draws = [d for d in draws if d['sorteo'] < target_s]
    train_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    
    all_target_draws = [d for d in draws if d['sorteo'] == target_s]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in all_target_draws}
    real_trad = real_by_mod.get('Tradicional', set())
    pozo_extra = set()
    for mod in ['Tradicional', 'Segunda', 'Revancha']:
        pozo_extra.update(real_by_mod.get(mod, set()))
        
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
    
    borda_pure = defaultdict(float)
    models_base = [(p_attn, 3.0), (p_t, w_rec), (p_cross, 1.5), (p_pcr, 1.4), (p_ddm, w_ddm)]
    for p_mod, w in models_base:
        rk = sorted(range(46), key=lambda x: p_mod[x], reverse=True)
        for rank, n in enumerate(rk):
            borda_pure[n] += w * (45 - rank)

    _, cooccur_mat = compute_exponential_cooccurrence_scores(train_trad, window=40, half_life=15.0)

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

    # Soliton Field (KdV)
    dens_circ = np.zeros(46)
    for d in train_trad[-15:]:
        for n in d['numbers']: dens_circ[n] += 1.0
    kernel_sech = np.array([1.0 / (math.cosh(dx/1.5)**2) for dx in range(-4, 5)])
    kernel_sech /= kernel_sech.sum()
    soliton_field = np.zeros(46)
    for n in range(46):
        s_val = sum(dens_circ[(n + dx) % 46] * kernel_sech[k_idx] for k_idx, dx in enumerate(range(-4, 5)))
        soliton_field[n] = s_val

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

    prev_draw_nums = train_trad[-1]['numbers']
    prev_set = set(prev_draw_nums)

    b_v = borda_pure.copy()
    if sat_decades:
        for n in range(46):
            if n // 10 in sat_decades: b_v[n] -= 50.0
    b_s = b_v.copy()
    for n in range(46): b_s[n] += 10.0 * soliton_field[n]

    rk_s = sorted(range(46), key=lambda x: b_s[x], reverse=True)
    t1_s, t2_s, t3_s, t4_s = rk_s[0:9], rk_s[9:18], rk_s[18:27], rk_s[27:36]
    rk_b = sorted(range(46), key=lambda x: b_v[x], reverse=True)
    t1_b, t2_b, t3_b, t4_b = rk_b[0:9], rk_b[9:18], rk_b[18:27], rk_b[27:36]

    def get_cands(t1_l, t2_l, t3_l, t4_l):
        cands = []
        for c_t1 in itertools.combinations(t1_l[:6], 3):
            for n2 in t2_l[:3]:
                for n3 in t3_l[:2]:
                    for n4 in t4_l[:2]:
                        cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                        if not es_ticket_valido(cand, min_distinct_decades=3): continue
                        qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                        if qr_c < 2 or qr_c > 4: continue
                        if len(set(cand) & prev_set) > 2: continue
                        cands.append(cand)
        if len(cands) < 3:
            cands = [c for c in itertools.combinations(t1_l[:6] + t2_l[:3], 6) if len(set(c) & prev_set) <= 2][:10]
        return cands

    cands_b1_base = get_cands(t1_s, t2_s, t3_s, t4_s)
    cands_b2_base = get_cands(t1_b, t2_b, t3_b, t4_b)
    cands_b3_base = get_cands(t1_b, t2_b, t3_b, t4_b)

    def sc_p23(c, bd):
        s = sum(bd[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
        for tr in itertools.combinations(c, 3): s += 16.0 * triplet_counts.get(tr, 0)
        bw_amp = (gamma_half**2) / ((sum(c) - E0_res)**2 + gamma_half**2)
        s += 28.0 * bw_amp
        n_ev = sum(1 for x in c if x % 2 == 0)
        if n_ev in (2, 3, 4): s += 8.0
        return s

    borda_d = b_v.copy()
    for n in range(46): borda_d[n] += 6.0 * counts_day[n]
    bd_b3 = b_v.copy()
    for n in range(46): bd_b3[n] += 6.0 * double_hits[n]

    b1 = max(cands_b1_base, key=lambda c: sc_p23(c, b_s))
    p12_base = set(itertools.combinations(b1, 2))

    for c_name, c_cfg in configs.items():
        w1 = c_cfg['b3_w1']
        w2 = c_cfg['b3_w2']
        w3 = c_cfg['b3_w3']
        b2_halo = c_cfg['b2_halo']

        cands_b2_filt = [c for c in cands_b2_base if len(set(c) & set(b1)) == target_ov and c != b1]
        if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_base if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
        
        if b2_halo:
            def b2_halo_bonus(cand):
                bonus = 0.0
                for n in cand:
                    min_d = min(min(abs(n - h), 46 - abs(n - h)) for h in b1)
                    if min_d == 1: bonus += 5.0
                return bonus
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d) + b2_halo_bonus(c)) if cands_b2_filt else cands_b2_base[0]
        else:
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2_base[0]

        p12 = p12_base | set(itertools.combinations(b2, 2))
        cands_b3_filt = [c for c in cands_b3_base if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
        if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_base if c != b1 and c != b2]

        def halo_bonus(cand):
            bonus = 0.0
            for n in cand:
                min_d = min(min(abs(n - h), 46 - abs(n - h)) for h in b1)
                if min_d == 1: bonus += w1
                elif min_d == 2: bonus += w2
                elif min_d == 3 and w3 > 0: bonus += w3
            return bonus

        b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12) + halo_bonus(c)) if cands_b3_filt else cands_b3_base[0]
        tickets = [b1, b2, b3]

        # Evaluación
        t_max_hits = max(len(set(t) & real_trad) for t in tickets)
        t_any_4 = t_max_hits >= 4
        t_any_3 = t_max_hits >= 3
        
        g_max_hits = 0
        pe_hits = set()
        for t in tickets:
            for mod, nums in real_by_mod.items():
                hits = len(set(t) & nums)
                if hits > g_max_hits: g_max_hits = hits
            pe_hits.update(set(t) & pozo_extra)
            
        g_any_4 = g_max_hits >= 4
        g_any_3 = g_max_hits >= 3
        pe_5 = len(pe_hits) >= 5
        seco = (g_max_hits < 3 and len(pe_hits) < 5)

        if t_any_4: results[c_name]['trad_4'] += 1
        if g_any_4: results[c_name]['glob_4'] += 1
        if t_any_3: results[c_name]['trad_3'] += 1
        if g_any_3: results[c_name]['glob_3'] += 1
        if pe_5: results[c_name]['pe_5'] += 1
        if seco: results[c_name]['secos'] += 1

    if (idx_eval + 1) % 110 == 0 or (idx_eval + 1) == n_eval:
        elapsed = time.time() - start_time
        max_t4 = max(results[c]['trad_4'] for c in configs)
        max_g4 = max(results[c]['glob_4'] for c in configs)
        print(f"[{idx_eval+1}/{n_eval}] sorteos evaluados ({elapsed:.1f}s) | Máx Trad 4+: {max_t4} | Glob 4+: {max_g4}")

print("="*115)
print(f"RESULTADOS FINALES: CALIBRACIÓN FINA HALO ORBITAL ({n_eval} SORTEOS / 1.980 BOLETOS)")
print("="*115)
header = f"{'Configuración de Pesos Halo':<38} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}"
print(header)
print("-" * len(header))
for c_name in configs:
    r = results[c_name]
    print(f"{c_name:<38} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print("="*115)

