import os, sys, itertools, time, math
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

# ===========================================================================
# EXPERIMENTO DE SÚPER-POTENCIACIÓN POR SINERGIA DE PORTFOLIO
# ===========================================================================
# S1: P23 Champion Ref (6 Trad 4+, 21 Glob 4+, 157 Glob 3+, 416 Secos)
# S2: Tri-Sinergia (B1 P23, B2 P23, B3 Multi-Tensor B4 -> buscando 21+ Glob 4 y 165+ Glob 3)
# S3: Tri-Sinergia con Cinemática en B1 y B2, y Multi-Tensor en B3
# S4: Súper-Sinergia con Calibración de Solapamiento Asimétrico

configs = {
    '0. P23 Champion Ref':                        {'mode': 'p23_ref'},
    'S2: Tri-Sinergia (B1/B2 P23 + B3 Tensor B4)':{'mode': 's2_tri_synergy'},
    'S3: Tri-Sinergia Cinemática Completa':       {'mode': 's3_kinematic_synergy'},
    'S4: Súper-Sinergia con Micro-PMI en B3':     {'mode': 's4_micro_pmi_b3'}
}

results = {c: {'trad_4': 0, 'glob_4': 0, 'trad_3': 0, 'glob_3': 0, 'pe_5': 0, 'secos': 0} for c in configs}

print("="*115)
print(f"EXPERIMENTO DE SÚPER-POTENCIACIÓN: SINERGIA DE 3 BOLETOS ({n_eval} SORTEOS / 1.980 BOLETOS)")
print("Objetivo: Mantener 6 Trad 4+ y 21 Glob 4+ mientras elevamos Glob 3+ hacia 165+ y bajamos secos...")
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

    # Difusión de Calor
    heat_source = p_t.copy()
    diffused_heat = np.zeros(46)
    deg = np.sum(cooccur_mat, axis=1) + 1e-5
    for u in range(46):
        diffused_heat[u] = sum((cooccur_mat[u, v] / math.sqrt(deg[u] * deg[v])) * heat_source[v] for v in range(46))
    diffused_heat /= (diffused_heat.max() + 1e-6)

    # Aceleración cinemática
    delay_accel = np.zeros(46)
    if len(train_trad) >= 25:
        sub_d = train_trad[-25:]
        delays = np.zeros((3, 46))
        for step_idx, step_back in enumerate([0, 5, 10]):
            d_slice = sub_d[:len(sub_d)-step_back] if step_back > 0 else sub_d
            last_seen = {n: len(d_slice) for n in range(46)}
            for idx_d, d_item in enumerate(d_slice):
                for num_item in d_item['numbers']:
                    last_seen[num_item] = len(d_slice) - 1 - idx_d
            for n in range(46):
                delays[step_idx, n] = last_seen[n]
        for n in range(46):
            v1 = delays[0, n] - delays[1, n]
            v2 = delays[1, n] - delays[2, n]
            accel = v1 - v2
            if delays[0, n] >= 12 and accel <= -2:
                delay_accel[n] = 1.0

    borda_kine = borda_pure.copy()
    for n in range(46):
        borda_kine[n] += 10.0 * delay_accel[n]

    # Borda Multi-Tensor para B3
    borda_multi = borda_pure.copy()
    for n in range(46):
        borda_multi[n] += 12.0 * diffused_heat[n] + 10.0 * delay_accel[n]

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

    # Triplet Tensor
    triplet_counts = Counter()
    for d in train_trad[-40:]:
        for tr in itertools.combinations(sorted(d['numbers']), 3):
            triplet_counts[tr] += 1

    # Rényi Collision Entropy
    p_renyi = (p_t + 1e-5) / (p_t.sum() + 46*1e-5)
    renyi_entropy = -math.log(np.sum(p_renyi**2) + 1e-6)
    target_ov = 4 if renyi_entropy < 3.55 else 3

    # Breit-Wigner Parameters
    E0_res = 138.0
    gamma_half = 15.0

    # Micro-EPR Cross-Modal Double Hits
    double_hits = Counter()
    for s_idx in range(target_s - 30, target_s):
        s_draws = [d for d in train_draws if d['sorteo'] == s_idx]
        if len(s_draws) >= 2:
            all_s_nums = Counter(n for d in s_draws for n in d['numbers'])
            for n, cnt in all_s_nums.items():
                if cnt >= 2: double_hits[n] += 1

    prev_draw_nums = train_trad[-1]['numbers']
    prev_set = set(prev_draw_nums)

    # Base generator helper with optimal 240-candidate compressed subspace
    def get_cands_base(t1_l, t2_l, t3_l, t4_l):
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
        if len(cands) < 3: cands = list(itertools.combinations(t1_l[:6] + t2_l[:3], 6))[:10]
        return cands

    def sc_p23(c, bd):
        s = sum(bd[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
        for tr in itertools.combinations(c, 3): s += 16.0 * triplet_counts.get(tr, 0)
        bw_amp = (gamma_half**2) / ((sum(c) - E0_res)**2 + gamma_half**2)
        s += 28.0 * bw_amp
        n_ev = sum(1 for x in c if x % 2 == 0)
        if n_ev in (2, 3, 4): s += 8.0
        return s

    for c_name, c_cfg in configs.items():
        mode = c_cfg['mode']

        if mode == 'p23_ref':
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

            cands_b1 = get_cands_base(t1_s, t2_s, t3_s, t4_s)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, b_s))

            cands_b2 = get_cands_base(t1_b, t2_b, t3_b, t4_b)
            borda_d = b_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]

            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands_base(t1_b, t2_b, t3_b, t4_b)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = b_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 's2_tri_synergy':
            # B1 & B2: P23 Clásico (cazadores de 4+ en Tradicional y Domingo)
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

            cands_b1 = get_cands_base(t1_s, t2_s, t3_s, t4_s)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, b_s))

            cands_b2 = get_cands_base(t1_b, t2_b, t3_b, t4_b)
            borda_d = b_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]

            # B3: Multi-Tensor B4 (inyecta difusión de calor + cinemática para capturar el pico de 3+ y reducir secos)
            b_m = borda_multi.copy()
            if sat_decades:
                for n in range(46):
                    if n // 10 in sat_decades: b_m[n] -= 50.0
            rk_m = sorted(range(46), key=lambda x: b_m[x], reverse=True)
            t1_m, t2_m, t3_m, t4_m = rk_m[0:9], rk_m[9:18], rk_m[18:27], rk_m[27:36]

            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands_base(t1_m, t2_m, t3_m, t4_m)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = b_m.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 's3_kinematic_synergy':
            # Cinemática en B1 y B2, y Multi-Tensor en B3
            b_k = borda_kine.copy()
            if sat_decades:
                for n in range(46):
                    if n // 10 in sat_decades: b_k[n] -= 50.0
            b_s = b_k.copy()
            for n in range(46): b_s[n] += 10.0 * soliton_field[n]
            
            rk_s = sorted(range(46), key=lambda x: b_s[x], reverse=True)
            t1_s, t2_s, t3_s, t4_s = rk_s[0:9], rk_s[9:18], rk_s[18:27], rk_s[27:36]
            rk_b = sorted(range(46), key=lambda x: b_k[x], reverse=True)
            t1_b, t2_b, t3_b, t4_b = rk_b[0:9], rk_b[9:18], rk_b[18:27], rk_b[27:36]

            cands_b1 = get_cands_base(t1_s, t2_s, t3_s, t4_s)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, b_s))

            cands_b2 = get_cands_base(t1_b, t2_b, t3_b, t4_b)
            borda_d = b_k.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]

            b_m = borda_multi.copy()
            if sat_decades:
                for n in range(46):
                    if n // 10 in sat_decades: b_m[n] -= 50.0
            rk_m = sorted(range(46), key=lambda x: b_m[x], reverse=True)
            t1_m, t2_m, t3_m, t4_m = rk_m[0:9], rk_m[9:18], rk_m[18:27], rk_m[27:36]

            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands_base(t1_m, t2_m, t3_m, t4_m)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = b_m.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 's4_micro_pmi_b3':
            # Micro-PMI en la función de scoring de B3
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

            cands_b1 = get_cands_base(t1_s, t2_s, t3_s, t4_s)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, b_s))

            cands_b2 = get_cands_base(t1_b, t2_b, t3_b, t4_b)
            borda_d = b_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]

            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands_base(t1_b, t2_b, t3_b, t4_b)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = b_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n] + 8.0 * delay_accel[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        hits_trad = [len(set(b) & real_trad) for b in tickets]
        max_t = max(hits_trad)
        if max_t >= 4: results[c_name]['trad_4'] += 1
        if max_t >= 3: results[c_name]['trad_3'] += 1
        if max_t <= 1: results[c_name]['secos'] += 1
        
        g4, g3 = False, False
        for mod, m_nums in real_by_mod.items():
            m_hits = [len(set(b) & m_nums) for b in tickets]
            if max(m_hits) >= 4: g4 = True
            if max(m_hits) >= 3: g3 = True
        if g4: results[c_name]['glob_4'] += 1
        if g3: results[c_name]['glob_3'] += 1
        
        pe_hits = [len(set(b) & pozo_extra) for b in tickets]
        if max(pe_hits) >= 5: results[c_name]['pe_5'] += 1

    if (idx_eval + 1) % 110 == 0:
        elapsed = time.time() - start_time
        print(f"[{idx_eval + 1}/{n_eval}] sorteos evaluados ({elapsed:.1f}s) | Máx Trad 4+: {max(r['trad_4'] for r in results.values())} | Glob 4+: {max(r['glob_4'] for r in results.values())}")

print('='*115)
print('RESULTADOS FINALES: SÚPER-POTENCIACIÓN POR SINERGIA (660 SORTEOS / 1.980 BOLETOS)')
print('='*115)
print(f"{'Modelo / Estrategia Sinergia':<46} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print('-'*115)
for c, r in results.items():
    print(f"{c:<46} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print('='*115)

