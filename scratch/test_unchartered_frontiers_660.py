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
# EXPERIMENTO DE FRONTERAS INEXPLORADAS (660 SORTEOS / 1.980 BOLETOS)
# ===========================================================================
configs = {
    '0. P23 Champion Ref (Greedy)':               {'mode': 'p23_ref'},
    'F1: Optimizador Simultáneo Conjunto (MIP)':  {'mode': 'f1_joint_mip'},
    'F2: Centralidad Supra-Laplaciana Multiplex': {'mode': 'f2_supra_laplacian'},
    'F3: Descomposición en Modos de Koopman (DMD)':{'mode': 'f3_koopman_dmd'},
    'F4: Modelo de Criticidad Auto-Organizada SOC':{'mode': 'f4_soc_sandpile'},
    'F5: Súper-Fusión de Frontera (MIP + Koopman)':{'mode': 'f5_frontier_supreme'}
}

results = {c: {'trad_4': 0, 'glob_4': 0, 'trad_3': 0, 'glob_3': 0, 'pe_5': 0, 'secos': 0} for c in configs}

print("="*115)
print(f"EXPERIMENTO DE FRONTERAS INEXPLORADAS ({n_eval} SORTEOS / 1.980 BOLETOS)")
print("Probando Optimizador Simultáneo MIP, Redes Multiplex 4-Capas, Modos de Koopman y SOC Sandpile...")
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

    # -----------------------------------------------------------------------
    # F2: Centralidad Supra-Laplaciana en Redes Multiplex (4 Capas)
    # -----------------------------------------------------------------------
    # Construye la red multicapa acoplando Tradicional, Segunda, Revancha y SiempreSale
    multiplex_scores = np.zeros(46)
    for mod_name, mod_weight in [('Tradicional', 1.0), ('Segunda', 0.8), ('Revancha', 0.7), ('SiempreSale', 0.6)]:
        m_draws = [d for d in train_draws[-60:] if d['modalidad'] == mod_name]
        if m_draws:
            for d in m_draws[-15:]:
                for n in d['numbers']:
                    multiplex_scores[n] += mod_weight
    multiplex_scores /= (multiplex_scores.max() + 1e-6)

    # -----------------------------------------------------------------------
    # F3: Descomposición en Modos de Koopman (Hankel DMD 1D por bolilla)
    # -----------------------------------------------------------------------
    # Matriz de series de tiempo binarias de los últimos 30 sorteos: X_0 y X_1
    koopman_persistent_energy = np.zeros(46)
    if len(train_trad) >= 30:
        time_series = np.zeros((46, 25))
        for t_step in range(25):
            d_t = train_trad[-25 + t_step]
            for num in d_t['numbers']:
                time_series[num, t_step] = 1.0
        X0 = time_series[:, :-1]
        X1 = time_series[:, 1:]
        # Aproximación de operador de Koopman K = X1 * pinv(X0)
        try:
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            S_inv = np.diag([1.0 / s if s > 1e-4 else 0.0 for s in S])
            A_koopman = X1 @ Vt.T @ S_inv @ U.T
            eigvals, eigvecs = np.linalg.eig(A_koopman)
            # Modos resonantes neutros (|lambda| cerca de 1)
            for mode_idx, lam in enumerate(eigvals):
                if 0.90 <= abs(lam) <= 1.10: # Modo cuasi-periódico persistente
                    mode_power = np.abs(eigvecs[:, mode_idx])
                    koopman_persistent_energy += mode_power
            koopman_persistent_energy /= (koopman_persistent_energy.max() + 1e-6)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # F4: Modelo de Criticidad Auto-Organizada (SOC Sandpile Avalanche)
    # -----------------------------------------------------------------------
    # Cada sorteo agrega 1 grano de arena a cada bolilla no extraída.
    # Cuando el atraso supera el umbral crítico z_c=12, acumula estrés avalanche.
    sandpile_stress = np.zeros(46)
    last_seen_soc = {n: len(train_trad) for n in range(46)}
    for idx_d, d_item in enumerate(train_trad):
        for num_item in d_item['numbers']:
            last_seen_soc[num_item] = len(train_trad) - 1 - idx_d
    for n in range(46):
        del_soc = last_seen_soc[n]
        if del_soc >= 12:
            # Estrés no-lineal de avalanche: toppling probability
            sandpile_stress[n] = 1.0 / (1.0 + math.exp(-0.35 * (del_soc - 15)))

    # Generator helper
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
    cands_b2 = get_cands_base(t1_b, t2_b, t3_b, t4_b)
    cands_b3 = get_cands_base(t1_b, t2_b, t3_b, t4_b)

    for c_name, c_cfg in configs.items():
        mode = c_cfg['mode']

        if mode == 'p23_ref':
            # Greedy clásico B1 -> B2 -> B3
            b1 = max(cands_b1, key=lambda c: sc_p23(c, b_s))
            borda_d = b_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]

            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = b_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'f1_joint_mip':
            # F1: OPTIMIZACIÓN SIMULTÁNEA CONJUNTA (JOINT TRILATERAL MIP OPTIMIZATION)
            # En lugar de greedy, evalúa los mejores N candidatos conjuntamente
            # maximizando: Fitness(B1) + Fitness(B2) + Fitness(B3) + 35 * ParesÚnicos(B1, B2, B3)
            top_b1 = sorted(cands_b1, key=lambda c: sc_p23(c, b_s), reverse=True)[:12]
            borda_d = b_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            top_b2 = sorted(cands_b2, key=lambda c: sc_p23(c, borda_d), reverse=True)[:15]
            bd_b3 = b_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            top_b3 = sorted(cands_b3, key=lambda c: sc_p23(c, bd_b3), reverse=True)[:15]

            best_triplet = None
            best_joint_score = -1e9

            for cand1 in top_b1:
                sc1 = sc_p23(cand1, b_s)
                set1 = set(cand1)
                pairs1 = set(itertools.combinations(cand1, 2))
                for cand2 in top_b2:
                    if cand2 == cand1: continue
                    ov12 = len(set1 & set(cand2))
                    if ov12 != target_ov: continue
                    sc2 = sc_p23(cand2, borda_d)
                    pairs12 = pairs1 | set(itertools.combinations(cand2, 2))
                    set2 = set(cand2)
                    for cand3 in top_b3:
                        if cand3 == cand1 or cand3 == cand2: continue
                        ov13 = len(set1 & set(cand3))
                        ov23 = len(set2 & set(cand3))
                        if not (2 <= ov13 <= 4 and 2 <= ov23 <= 4): continue
                        sc3 = sc_p23(cand3, bd_b3)
                        pairs123 = pairs12 | set(itertools.combinations(cand3, 2))
                        # Función objetivo conjunta
                        joint_score = sc1 + sc2 + sc3 + 30.0 * len(pairs123)
                        if joint_score > best_joint_score:
                            best_joint_score = joint_score
                            best_triplet = [cand1, cand2, cand3]
            tickets = best_triplet if best_triplet else [top_b1[0], top_b2[0], top_b3[0]]

        elif mode == 'f2_supra_laplacian':
            # F2: Centralidad Supra-Laplaciana en Redes Multiplex
            b_multi = b_v.copy()
            for n in range(46): b_multi[n] += 15.0 * multiplex_scores[n]
            rk_m = sorted(range(46), key=lambda x: b_multi[x], reverse=True)
            t1_m, t2_m, t3_m, t4_m = rk_m[0:9], rk_m[9:18], rk_m[18:27], rk_m[27:36]

            cands_b1_m = get_cands_base(t1_s, t2_s, t3_s, t4_s)
            b1 = max(cands_b1_m, key=lambda c: sc_p23(c, b_s))

            cands_b2_m = get_cands_base(t1_m, t2_m, t3_m, t4_m)
            borda_d = b_multi.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2_m if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_m if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2_m[0]

            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_m = get_cands_base(t1_m, t2_m, t3_m, t4_m)
            cands_b3_filt = [c for c in cands_b3_m if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_m if c != b1 and c != b2]
            bd_b3 = b_multi.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3_m[0]
            tickets = [b1, b2, b3]

        elif mode == 'f3_koopman_dmd':
            # F3: Modos Armónicos de Koopman
            b_koop = b_v.copy()
            for n in range(46): b_koop[n] += 16.0 * koopman_persistent_energy[n]
            rk_k = sorted(range(46), key=lambda x: b_koop[x], reverse=True)
            t1_k, t2_k, t3_k, t4_k = rk_k[0:9], rk_k[9:18], rk_k[18:27], rk_k[27:36]

            cands_b1_k = get_cands_base(t1_s, t2_s, t3_s, t4_s)
            b1 = max(cands_b1_k, key=lambda c: sc_p23(c, b_s))

            cands_b2_k = get_cands_base(t1_k, t2_k, t3_k, t4_k)
            borda_d = b_koop.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2_k if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_k if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2_k[0]

            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_k = get_cands_base(t1_k, t2_k, t3_k, t4_k)
            cands_b3_filt = [c for c in cands_b3_k if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_k if c != b1 and c != b2]
            bd_b3 = b_koop.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3_k[0]
            tickets = [b1, b2, b3]

        elif mode == 'f4_soc_sandpile':
            # F4: SOC Sandpile Avalanche
            b_soc = b_v.copy()
            for n in range(46): b_soc[n] += 14.0 * sandpile_stress[n]
            rk_soc = sorted(range(46), key=lambda x: b_soc[x], reverse=True)
            t1_soc, t2_soc, t3_soc, t4_soc = rk_soc[0:9], rk_soc[9:18], rk_soc[18:27], rk_soc[27:36]

            cands_b1_soc = get_cands_base(t1_s, t2_s, t3_s, t4_s)
            b1 = max(cands_b1_soc, key=lambda c: sc_p23(c, b_s))

            cands_b2_soc = get_cands_base(t1_soc, t2_soc, t3_soc, t4_soc)
            borda_d = b_soc.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2_soc if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_soc if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2_soc[0]

            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_soc = get_cands_base(t1_soc, t2_soc, t3_soc, t4_soc)
            cands_b3_filt = [c for c in cands_b3_soc if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_soc if c != b1 and c != b2]
            bd_b3 = b_soc.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3_soc[0]
            tickets = [b1, b2, b3]

        elif mode == 'f5_frontier_supreme':
            # F5: Súper-Fusión: Koopman en Borda + Optimización Simultánea Conjunta MIP
            b_sup = b_v.copy()
            for n in range(46): b_sup[n] += 10.0 * koopman_persistent_energy[n] + 8.0 * multiplex_scores[n]
            rk_sup = sorted(range(46), key=lambda x: b_sup[x], reverse=True)
            t1_sup, t2_sup, t3_sup, t4_sup = rk_sup[0:9], rk_sup[9:18], rk_sup[18:27], rk_sup[27:36]

            c_b1 = get_cands_base(t1_s, t2_s, t3_s, t4_s)
            c_b2 = get_cands_base(t1_sup, t2_sup, t3_sup, t4_sup)
            c_b3 = get_cands_base(t1_sup, t2_sup, t3_sup, t4_sup)

            top_b1 = sorted(c_b1, key=lambda c: sc_p23(c, b_s), reverse=True)[:12]
            borda_d = b_sup.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            top_b2 = sorted(c_b2, key=lambda c: sc_p23(c, borda_d), reverse=True)[:15]
            bd_b3 = b_sup.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            top_b3 = sorted(c_b3, key=lambda c: sc_p23(c, bd_b3), reverse=True)[:15]

            best_triplet = None
            best_joint_score = -1e9
            for cand1 in top_b1:
                sc1 = sc_p23(cand1, b_s)
                set1 = set(cand1)
                pairs1 = set(itertools.combinations(cand1, 2))
                for cand2 in top_b2:
                    if cand2 == cand1: continue
                    ov12 = len(set1 & set(cand2))
                    if ov12 != target_ov: continue
                    sc2 = sc_p23(cand2, borda_d)
                    pairs12 = pairs1 | set(itertools.combinations(cand2, 2))
                    set2 = set(cand2)
                    for cand3 in top_b3:
                        if cand3 == cand1 or cand3 == cand2: continue
                        ov13 = len(set1 & set(cand3))
                        ov23 = len(set2 & set(cand3))
                        if not (2 <= ov13 <= 4 and 2 <= ov23 <= 4): continue
                        sc3 = sc_p23(cand3, bd_b3)
                        pairs123 = pairs12 | set(itertools.combinations(cand3, 2))
                        joint_score = sc1 + sc2 + sc3 + 30.0 * len(pairs123)
                        if joint_score > best_joint_score:
                            best_joint_score = joint_score
                            best_triplet = [cand1, cand2, cand3]
            tickets = best_triplet if best_triplet else [top_b1[0], top_b2[0], top_b3[0]]

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
print('RESULTADOS FINALES: FRONTERAS INEXPLORADAS (660 SORTEOS / 1.980 BOLETOS)')
print('='*115)
print(f"{'Modelo / Teoría Inexplorada':<46} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print('-'*115)
for c, r in results.items():
    print(f"{c:<46} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print('='*115)

