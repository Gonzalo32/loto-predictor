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
# EXPERIMENTO DE EXPANSIÓN DE CAPACIDAD COMBINATORIA Y DE INFORMACIÓN
# ===========================================================================
configs = {
    '0. P23 Champion Ref (21 Glob 4+, 6 Trad 4+)': {'mode': 'p23_ref'},
    'C1: Estratificación Elástica Asimétrica':      {'mode': 'c1_elastic_strata'},
    'C2: Max-Capacity Quadruplet Packing':          {'mode': 'c2_quad_capacity'},
    'C3: Canal de Información Inter-Modal':         {'mode': 'c3_channel_info'},
    'C4: Expansión Adaptativa de Tier 1 (T1[:8])':  {'mode': 'c4_expanded_t1'},
    'C5: Súper-Fusión de Capacidad Total':          {'mode': 'c5_super_capacity'}
}

results = {c: {'trad_4': 0, 'glob_4': 0, 'trad_3': 0, 'glob_3': 0, 'pe_5': 0, 'secos': 0} for c in configs}

print("="*115)
print(f"EXPERIMENTO: EXPANSIÓN DE CAPACIDAD ({n_eval} SORTEOS / 1.980 BOLETOS)")
print("Probando Estratificación Elástica Asimétrica, Max-Capacity Quadruplets, Canal Inter-Modal y T1 Expandido...")
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
    borda_sol = borda_v.copy()
    for n in range(46): borda_sol[n] += 10.0 * soliton_field[n]
    rk_sol = sorted(range(46), key=lambda x: borda_sol[x], reverse=True)
    t1_sol, t2_sol, t3_sol, t4_sol = rk_sol[0:9], rk_sol[9:18], rk_sol[18:27], rk_sol[27:36]

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

    # Base generator helper with persistence filter
    def get_cands_custom(t1_l, t2_l, t3_l, t4_l, strata=(3, 1, 1, 1), t1_depth=6):
        n_t1, n_t2, n_t3, n_t4 = strata
        cands = []
        for c_t1 in itertools.combinations(t1_l[:t1_depth], n_t1):
            for c_t2 in itertools.combinations(t2_l[:4], n_t2):
                for c_t3 in itertools.combinations(t3_l[:3], n_t3):
                    for c_t4 in itertools.combinations(t4_l[:3], n_t4):
                        cand = tuple(sorted(list(c_t1) + list(c_t2) + list(c_t3) + list(c_t4)))
                        if not es_ticket_valido(cand, min_distinct_decades=3): continue
                        qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                        if qr_c < 2 or qr_c > 4: continue
                        if len(set(cand) & prev_set) > 2: continue
                        cands.append(cand)
        if len(cands) < 3:
            cands = [c for c in itertools.combinations(t1_l[:6] + t2_l[:3], 6) if es_ticket_valido(c, min_distinct_decades=3)]
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
            cands_b1 = get_cands_custom(t1_sol, t2_sol, t3_sol, t4_sol, strata=(3, 1, 1, 1), t1_depth=6)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, borda_sol))
            
            cands_b2 = get_cands_custom(t1, t2, t3, t4, strata=(3, 1, 1, 1), t1_depth=6)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands_custom(t1, t2, t3, t4, strata=(3, 1, 1, 1), t1_depth=6)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'c1_elastic_strata':
            # ESTRATIFICACIÓN ELÁSTICA ASIMÉTRICA:
            # B1: Concentrado (4 de T1_sol, 1 de T2_sol, 1 de T3_sol, 0 de T4)
            cands_b1 = get_cands_custom(t1_sol, t2_sol, t3_sol, t4_sol, strata=(4, 1, 1, 0), t1_depth=6)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, borda_sol)) if cands_b1 else max(get_cands_custom(t1_sol, t2_sol, t3_sol, t4_sol), key=lambda c: sc_p23(c, borda_sol))
            
            # B2: Puente Balanceado (3 de T1, 2 de T2, 1 de T3, 0 de T4)
            cands_b2 = get_cands_custom(t1, t2, t3, t4, strata=(3, 2, 1, 0), t1_depth=6)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            # B3: Cobertura Dispersa (2 de T1, 2 de T2, 1 de T3, 1 de T4)
            cands_b3 = get_cands_custom(t1, t2, t3, t4, strata=(2, 2, 1, 1), t1_depth=6)
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'c2_quad_capacity':
            # Maximización de Capacidad de Cuádruplas: B3 busca activamente cuádruplas
            # que no hayan sido cubiertas por B1 y B2 pero que tengan alta co-ocurrencia
            cands_b1 = get_cands_custom(t1_sol, t2_sol, t3_sol, t4_sol, strata=(3, 1, 1, 1), t1_depth=6)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, borda_sol))
            
            cands_b2 = get_cands_custom(t1, t2, t3, t4, strata=(3, 1, 1, 1), t1_depth=6)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands_custom(t1, t2, t3, t4, strata=(3, 1, 1, 1), t1_depth=6)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            # Bonus por cuádruplas descubiertas de alta afinidad
            def sc_quad_bonus(c):
                s = sc_p23(c, bd_b3) + 25.0 * len(set(itertools.combinations(c, 2)) - p12)
                new_quads = len(set(itertools.combinations(c, 4)) - q12)
                return s + 2.5 * new_quads
            b3 = max(cands_b3_filt, key=sc_quad_bonus) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'c3_channel_info':
            # Capacidad de Canal Inter-Modal: inyecta peso de información mutua
            # a las bolillas con alta conductividad entre Tradicional y las otras 3 modalidades
            cross_modal_conductance = np.zeros(46)
            for d in train_draws[-50:]:
                for n in d['numbers']: cross_modal_conductance[n] += 1.0
            cross_modal_conductance /= (cross_modal_conductance.max() + 1e-5)
            
            borda_cond = borda_v.copy()
            for n in range(46): borda_cond[n] += 25.0 * cross_modal_conductance[n]
            rk_c = sorted(range(46), key=lambda x: borda_cond[x], reverse=True)
            t1_c, t2_c, t3_c, t4_c = rk_c[0:9], rk_c[9:18], rk_c[18:27], rk_c[27:36]

            cands_b1 = get_cands_custom(t1_sol, t2_sol, t3_sol, t4_sol, strata=(3, 1, 1, 1), t1_depth=6)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, borda_sol))
            
            cands_b2 = get_cands_custom(t1_c, t2_c, t3_c, t4_c, strata=(3, 1, 1, 1), t1_depth=6)
            borda_d = borda_cond.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands_custom(t1_c, t2_c, t3_c, t4_c, strata=(3, 1, 1, 1), t1_depth=6)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = borda_cond.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'c4_expanded_t1':
            # Expansión de Capacidad de Tier 1 a los primeros 8 números élite (t1_depth=8)
            cands_b1 = get_cands_custom(t1_sol, t2_sol, t3_sol, t4_sol, strata=(3, 1, 1, 1), t1_depth=8)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, borda_sol))
            
            cands_b2 = get_cands_custom(t1, t2, t3, t4, strata=(3, 1, 1, 1), t1_depth=8)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands_custom(t1, t2, t3, t4, strata=(3, 1, 1, 1), t1_depth=8)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p23(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'c5_super_capacity':
            # Fusión de Capacidad: T1 Depth=7 + Quad Packing en B3
            cands_b1 = get_cands_custom(t1_sol, t2_sol, t3_sol, t4_sol, strata=(3, 1, 1, 1), t1_depth=7)
            b1 = max(cands_b1, key=lambda c: sc_p23(c, borda_sol))
            
            cands_b2 = get_cands_custom(t1, t2, t3, t4, strata=(3, 1, 1, 1), t1_depth=7)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p23(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands_custom(t1, t2, t3, t4, strata=(3, 1, 1, 1), t1_depth=7)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            def sc_c5(c):
                s = sc_p23(c, bd_b3) + 28.0 * len(set(itertools.combinations(c, 2)) - p12)
                s += 2.0 * len(set(itertools.combinations(c, 4)) - q12)
                return s
            b3 = max(cands_b3_filt, key=sc_c5) if cands_b3_filt else cands_b3[0]
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
        print(f"[{idx_eval + 1}/{n_eval}] sorteos evaluados ({elapsed:.1f}s) | Máx Trad 4+: {max(r['trad_4'] for r in results.values())} | Máx Glob 4+: {max(r['glob_4'] for r in results.values())}")

print('='*115)
print('RESULTADOS FINALES: EXPANSIÓN DE CAPACIDAD (660 SORTEOS / 1.980 BOLETOS)')
print('='*115)
print(f"{'Modelo / Estrategia de Capacidad':<46} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print('-'*115)
for c, r in results.items():
    print(f"{c:<46} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print('='*115)

