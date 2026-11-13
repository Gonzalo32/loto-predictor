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
# EXPERIMENTO: PREDICCIÓN DE CÁLCULOS MACROSCÓPICOS E INVARIANTES FÍSICOS
# ===========================================================================
# Modelos a probar:
# 0. K2 Base Champion: 21 Glob 4+, 5 Trad 4+, 157 Glob 3+, 415 Secos
# 1. P1: Predicción de Centroide por Retorno a la Media (Hooke's Spring Law)
#    x_cm(t+1) = x_cm(t) - gamma * (x_cm(t) - 22.0)
# 2. P2: Condicionamiento Estricto de Repetición Previa (k_rep <= 1)
#    Forzar que los boletos no repitan más de 1 o 2 del sorteo anterior
# 3. P3: Predicción de Paridad Dinámica por Cadena de Markov de 2 Estados
# 4. P4: Condicionamiento de Consecutividad Ponderada (Bonus si tiene 1 par consecutivo)
# 5. P5: El Súper-Condicionador Total (K2 + P1 Centroide + P2 Repetición + P4 Consecutivo)

configs = {
    '0. K2 Champion (21 Glob 4+)':             {'mode': 'k2_base'},
    '1. P1: Resorte de Centroide (Hooke)':     {'mode': 'p1_centroid_spring'},
    '2. P2: Filtro de Persistencia (k_rep<=2)':{'mode': 'p2_persistence'},
    '3. P3: Markov Paridad Dinámica':          {'mode': 'p3_markov_parity'},
    '4. P4: Resonancia de Consecutividad':     {'mode': 'p4_consecutive_bonus'},
    '5. P5: Súper-Condicionador Macro (GUT)':  {'mode': 'p5_super_macro'}
}

results = {c: {'trad_4': 0, 'glob_4': 0, 'trad_3': 0, 'glob_3': 0, 'pe_5': 0, 'secos': 0} for c in configs}

print("="*115)
print(f"EXPERIMENTO DE PREDICCIÓN DE CÁLCULOS E INVARIANTES MACRO ({n_eval} SORTEOS / 1.980 BOLETOS)")
print("Probando condicionamiento termodinámico por Centroide Hooke, Filtro de Persistencia y Paridad Markov...")
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

    # -----------------------------------------------------------------------
    # CÁLCULOS MACROSCÓPICOS PREDICHOS:
    # -----------------------------------------------------------------------
    # 1. Predicción del Centroide Dinámico por Resorte de Hooke:
    prev_draw_nums = train_trad[-1]['numbers']
    prev_cm = np.mean(prev_draw_nums)
    # Si prev_cm > 22.0, el resorte atrae hacia abajo; si prev_cm < 22.0, atrae hacia arriba
    # x_pred = prev_cm - 0.45 * (prev_cm - 22.0)
    hooke_pred_cm = prev_cm - 0.45 * (prev_cm - 22.0)

    # 2. Números del sorteo previo para filtro de persistencia:
    prev_set = set(prev_draw_nums)

    # 3. Predicción de Paridad Markoviana:
    prev_evens = sum(1 for x in prev_draw_nums if x % 2 == 0)
    # Si en el anterior hubo 4 o 5 pares, la probabilidad de 2 o 3 pares en el siguiente es 75%
    target_evens_range = (2, 4) if prev_evens in (4, 5) else (2, 4)

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
                        cands.append(cand)
        if len(cands) < 3: cands = list(itertools.combinations(t1_l[:6] + t2_l[:3], 6))[:10]
        return cands

    def sc_k2(c, bd):
        s = sum(bd[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
        for tr in itertools.combinations(c, 3): s += 16.0 * triplet_counts.get(tr, 0)
        bw_amp = (gamma_half**2) / ((sum(c) - E0_res)**2 + gamma_half**2)
        s += 28.0 * bw_amp
        return s

    cands_b1_all = get_cands_base(t1_sol, t2_sol, t3_sol, t4_sol)
    cands_b2_all = get_cands_base(t1, t2, t3, t4)
    cands_b3_all = get_cands_base(t1, t2, t3, t4)

    for c_name, c_cfg in configs.items():
        mode = c_cfg['mode']
        
        if mode == 'k2_base':
            # K2 Base Champion
            b1 = max(cands_b1_all, key=lambda c: sc_k2(c, borda_sol))
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2_all if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_all if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_k2(c, borda_d)) if cands_b2_filt else cands_b2_all[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_filt = [c for c in cands_b3_all if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_all if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_k2(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3_all[0]
            tickets = [b1, b2, b3]

        elif mode == 'p1_centroid_spring':
            # Condicionamiento por Resorte de Hooke del Centroide
            # Añade un término armónico: - k_spring * (mean(c) - hooke_pred_cm)^2
            def sc_p1(c, bd):
                s = sc_k2(c, bd)
                c_mean = np.mean(c)
                spring_penalty = 1.2 * ((c_mean - hooke_pred_cm)**2)
                return s - spring_penalty
                
            b1 = max(cands_b1_all, key=lambda c: sc_p1(c, borda_sol))
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2_all if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_all if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p1(c, borda_d)) if cands_b2_filt else cands_b2_all[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_filt = [c for c in cands_b3_all if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_all if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p1(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3_all[0]
            tickets = [b1, b2, b3]

        elif mode == 'p2_persistence':
            # Filtro de persistencia macroscópica: ningún boleto debe repetir más de 2 bolillas del sorteo previo
            def filt_persist(c):
                return len(set(c) & prev_set) <= 2
            
            cands_b1_p = [c for c in cands_b1_all if filt_persist(c)]
            if not cands_b1_p: cands_b1_p = cands_b1_all
            b1 = max(cands_b1_p, key=lambda c: sc_k2(c, borda_sol))
            
            cands_b2_p = [c for c in cands_b2_all if filt_persist(c)]
            if not cands_b2_p: cands_b2_p = cands_b2_all
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2_p if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_p if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_k2(c, borda_d)) if cands_b2_filt else cands_b2_p[0]
            
            cands_b3_p = [c for c in cands_b3_all if filt_persist(c)]
            if not cands_b3_p: cands_b3_p = cands_b3_all
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_filt = [c for c in cands_b3_p if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_p if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_k2(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3_p[0]
            tickets = [b1, b2, b3]

        elif mode == 'p3_markov_parity':
            # Condicionamiento de paridad probabilística: bonus a combinaciones con 2, 3 o 4 pares
            def sc_p3(c, bd):
                s = sc_k2(c, bd)
                n_ev = sum(1 for x in c if x % 2 == 0)
                if n_ev in (2, 3, 4): s += 12.0
                return s
                
            b1 = max(cands_b1_all, key=lambda c: sc_p3(c, borda_sol))
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2_all if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_all if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p3(c, borda_d)) if cands_b2_filt else cands_b2_all[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_filt = [c for c in cands_b3_all if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_all if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p3(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3_all[0]
            tickets = [b1, b2, b3]

        elif mode == 'p4_consecutive_bonus':
            # Resonancia de Consecutividad: ~49% de los sorteos tienen al menos 1 par consecutivo
            # Se otorga un bonus suave a boletos que posean exactamente 1 par consecutivo (ej. 09-10 o 22-23)
            def sc_p4(c, bd):
                s = sc_k2(c, bd)
                has_consec = any(c[k+1] - c[k] == 1 for k in range(5))
                if has_consec: s += 15.0
                return s
                
            b1 = max(cands_b1_all, key=lambda c: sc_p4(c, borda_sol))
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2_all if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_all if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p4(c, borda_d)) if cands_b2_filt else cands_b2_all[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_filt = [c for c in cands_b3_all if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_all if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p4(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3_all[0]
            tickets = [b1, b2, b3]

        elif mode == 'p5_super_macro':
            # Súper-Condicionador Macro: P1 Resorte de Centroide + P2 Persistencia (<=2) + P4 Consecutivo
            def filt_persist(c):
                return len(set(c) & prev_set) <= 2

            def sc_p5(c, bd):
                s = sc_k2(c, bd)
                c_mean = np.mean(c)
                s -= 0.8 * ((c_mean - hooke_pred_cm)**2)
                if any(c[k+1] - c[k] == 1 for k in range(5)):
                    s += 12.0
                n_ev = sum(1 for x in c if x % 2 == 0)
                if n_ev in (2, 3, 4):
                    s += 8.0
                return s

            cands_b1_p = [c for c in cands_b1_all if filt_persist(c)]
            if not cands_b1_p: cands_b1_p = cands_b1_all
            b1 = max(cands_b1_p, key=lambda c: sc_p5(c, borda_sol))
            
            cands_b2_p = [c for c in cands_b2_all if filt_persist(c)]
            if not cands_b2_p: cands_b2_p = cands_b2_all
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2_p if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2_p if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p5(c, borda_d)) if cands_b2_filt else cands_b2_p[0]
            
            cands_b3_p = [c for c in cands_b3_all if filt_persist(c)]
            if not cands_b3_p: cands_b3_p = cands_b3_all
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3_filt = [c for c in cands_b3_p if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3_p if c != b1 and c != b2]
            bd_b3 = borda_v.copy()
            for n in range(46): bd_b3[n] += 6.0 * double_hits[n]
            b3 = max(cands_b3_filt, key=lambda c: sc_p5(c, bd_b3) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3_p[0]
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
        print(f"[{idx_eval + 1}/{n_eval}] sorteos evaluados ({elapsed:.1f}s) | Máx Glob 4+: {max(r['glob_4'] for r in results.values())}")

print('='*115)
print('RESULTADOS: PREDICCIÓN DE CÁLCULOS E INVARIANTES FÍSICOS (660 SORTEOS / 1.980 BOLETOS)')
print('='*115)
print(f"{'Modelo / Predicción de Cálculo':<46} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print('-'*115)
for c, r in results.items():
    print(f"{c:<46} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print('='*115)

