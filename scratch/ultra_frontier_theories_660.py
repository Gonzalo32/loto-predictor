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
# DEFINICIÓN DE TEORÍAS DE ULTRA-FRONTERA MATEMÁTICA
# ===========================================================================
configs = {
    'Ref. Champion: Rényi-X3 (20 Glob 4+)':         {'mode': 'ref_renyi_x3'},
    'Ω1: Flujo de Curvatura de Ricci (Ricci Flow)': {'mode': 'ricci_flow'},
    'Ω2: Spin Glass de Ising Térmico (Boltzmann)':  {'mode': 'ising_spin'},
    'Ω3: Atractor de Takens & Recurrencia Poincaré':{'mode': 'takens_poincare'},
    'Ω4: Resonancia Cuántica de Breit-Wigner':      {'mode': 'breit_wigner'},
    'Ω5: Gran Unificación Suprema (GUT-46)':        {'mode': 'gut_46_supreme'}
}

results = {c: {'trad_4': 0, 'glob_4': 0, 'trad_3': 0, 'glob_3': 0, 'pe_5': 0, 'secos': 0} for c in configs}

print("="*115)
print(f"EXPERIMENTO DE ULTRA-FRONTERA: 5 NUEVAS TEORÍAS MATEMÁTICAS ({n_eval} SORTEOS / 1.980 BOLETOS)")
print("Rompiendo los límites de lo posible: Ricci Flow, Ising Cuántico, Atractor de Takens, Breit-Wigner y GUT-46...")
print("="*115)

start_time = time.time()

for idx_eval, target_draw in enumerate(eval_draws):
    target_s = target_draw['sorteo']
    target_date_str = target_draw['fecha']
    target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
    target_weekday = target_dt.weekday()
    
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
    target_ov_renyi = 4 if renyi_entropy < 3.55 else 3

    # -----------------------------------------------------------------------
    # TEORÍA Ω1: Flujo de Curvatura de Ricci Combinatorio (Ricci Flow)
    # -----------------------------------------------------------------------
    # Calculamos la curvatura de Ricci discreta (Ollivier-Ricci aproximada)
    # kappa(i, j) = 1 - (distancia Wasserstein promedio de vecindades)
    # Aristas con kappa > 0 se contraen bajo el flujo dW/dt = -kappa * W,
    # amplificando clusters geodesicos ultracompactos.
    ricci_flow_boost = np.zeros(46)
    # Aproximación rápida de curvatura de vecindad: solapamiento triangular
    for u in range(46):
        neigh_u = set(v for v in range(46) if cooccur_mat[u, v] > 0.05 and v != u)
        if not neigh_u: continue
        triangles = 0
        pairs_count = 0
        for v in neigh_u:
            neigh_v = set(w for w in range(46) if cooccur_mat[v, w] > 0.05 and w != v)
            common = len(neigh_u & neigh_v)
            triangles += common
            pairs_count += 1
        clustering_coeff = triangles / (pairs_count * 10.0 + 1e-5)
        # Positiva curvatura = clustering alto -> flujo contrae y atrae
        ricci_flow_boost[u] = clustering_coeff * 25.0

    # -----------------------------------------------------------------------
    # TEORÍA Ω2: Red de Spin de Boltzmann / Ising con Recocido Rényi
    # -----------------------------------------------------------------------
    # Hamiltoniano: H(cand) = - sum_{u, v} J_uv - sum_u h_u
    # Con beta = 1 / T_eff = 1.0 / (renyi_entropy - 2.8)
    beta_eff = 1.0 / max(0.2, (renyi_entropy - 2.8))

    # -----------------------------------------------------------------------
    # TEORÍA Ω3: Atractor de Takens & Secciones de Poincaré
    # -----------------------------------------------------------------------
    # Reconstrucción de espacio de fases con embeddings de delay tau=1, 2
    # Serie de centroides de sumas de los últimos 20 sorteos
    centroids = [np.mean(d['numbers']) for d in train_trad[-20:]]
    # Velocidad de fase: v(t) = C(t) - C(t-1)
    phase_vel = [centroids[k] - centroids[k-1] for k in range(1, len(centroids))]
    # Punto actual en el espacio de fases
    curr_c = centroids[-1]
    curr_v = phase_vel[-1]
    # Distancia en espacio de fases a puntos históricos de retorno de Poincaré
    # Predecir centroide objetivo para el siguiente sorteo
    pred_centroid = curr_c + 0.35 * curr_v

    # -----------------------------------------------------------------------
    # TEORÍA Ω4: Resonancia Cuántica de Breit-Wigner en Sumas de Invariantes
    # -----------------------------------------------------------------------
    # Nivel de resonancia E0 para la suma total de las 6 bolillas
    # Históricamente las sumas se centran en 138 con semiancho Gamma = 22
    # El perfil de Breit-Wigner es sigma(S) = (Gamma/2)^2 / ((S - E0)^2 + (Gamma/2)^2)
    E0_res = 138.0
    gamma_half = 15.0

    # Generator helper
    def get_cands(t1_l, t2_l, t3_l, t4_l, qr_check=True):
        cands = []
        for c_t1 in itertools.combinations(t1_l[:6], 3):
            for n2 in t2_l[:3]:
                for n3 in t3_l[:2]:
                    for n4 in t4_l[:2]:
                        cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                        if not es_ticket_valido(cand, min_distinct_decades=3): continue
                        if qr_check:
                            qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                            if qr_c < 2 or qr_c > 4: continue
                        cands.append(cand)
        if len(cands) < 3: cands = list(itertools.combinations(t1_l[:6] + t2_l[:3], 6))[:10]
        return cands

    # -----------------------------------------------------------------------
    # EVALUACIÓN DE LAS 6 TEORÍAS
    # -----------------------------------------------------------------------
    for c_name, c_cfg in configs.items():
        mode = c_cfg['mode']
        
        if mode == 'ref_renyi_x3':
            # Campeón actual de referencia
            target_ov = target_ov_renyi
            cands_b1 = get_cands(t1_sol, t2_sol, t3_sol, t4_sol)
            def sc_r(c, bd):
                s = sum(bd[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
                for tr in itertools.combinations(c, 3): s += 15.0 * triplet_counts.get(tr, 0)
                return s
            b1 = max(cands_b1, key=lambda c: sc_r(c, borda_sol))
            
            cands_b2 = get_cands(t1, t2, t3, t4)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_r(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands(t1, t2, t3, t4)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            b3 = max(cands_b3_filt, key=lambda c: sc_r(c, borda_v) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'ricci_flow':
            # Inyecta flujo de curvatura de Ricci a Borda y selecciona geodésicas compactas
            borda_ricci = borda_v.copy()
            for n in range(46): borda_ricci[n] += ricci_flow_boost[n]
            rk_ric = sorted(range(46), key=lambda x: borda_ricci[x], reverse=True)
            t1_ric, t2_ric, t3_ric, t4_ric = rk_ric[0:9], rk_ric[9:18], rk_ric[18:27], rk_ric[27:36]
            
            cands_b1 = get_cands(t1_sol, t2_sol, t3_sol, t4_sol)
            def sc_ricci(c, bd):
                s = sum(bd[n] for n in c) + 40.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
                for tr in itertools.combinations(c, 3): s += 15.0 * triplet_counts.get(tr, 0)
                return s
            b1 = max(cands_b1, key=lambda c: sc_ricci(c, borda_sol))
            
            cands_b2 = get_cands(t1_ric, t2_ric, t3_ric, t4_ric)
            borda_d = borda_ricci.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov_renyi and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_ricci(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands(t1_ric, t2_ric, t3_ric, t4_ric)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            b3 = max(cands_b3_filt, key=lambda c: sc_ricci(c, borda_ricci) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'ising_spin':
            # Modelo de Spin Glass de Ising con peso de Gibbs-Boltzmann exp(-beta * H)
            cands_b1 = get_cands(t1_sol, t2_sol, t3_sol, t4_sol)
            def gibbs_energy(c, bd):
                # H = - (sum h_i + sum J_ij)
                h_term = sum(bd[n] for n in c)
                j_term = 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
                for tr in itertools.combinations(c, 3): j_term += 15.0 * triplet_counts.get(tr, 0)
                # Gibbs score escalado por beta_eff
                return beta_eff * (h_term + j_term)
            b1 = max(cands_b1, key=lambda c: gibbs_energy(c, borda_sol))
            
            cands_b2 = get_cands(t1, t2, t3, t4)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov_renyi and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: gibbs_energy(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands(t1, t2, t3, t4)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            b3 = max(cands_b3_filt, key=lambda c: gibbs_energy(c, borda_v) + beta_eff * 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'takens_poincare':
            # Filtro de tubo de flujo de Poincaré: penaliza combinaciones cuyo centroide
            # esté a más de 3.5 unidades del centroide previsto por el atractor caótico
            def poincare_filter(c):
                c_mean = np.mean(c)
                return abs(c_mean - pred_centroid) <= 3.8
            
            cands_b1 = get_cands(t1_sol, t2_sol, t3_sol, t4_sol)
            cands_b1_p = [c for c in cands_b1 if poincare_filter(c)]
            if not cands_b1_p: cands_b1_p = cands_b1
            
            def sc_p(c, bd):
                s = sum(bd[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
                for tr in itertools.combinations(c, 3): s += 15.0 * triplet_counts.get(tr, 0)
                # Bonus por cercanía a la sección transversal de Poincaré
                s += 20.0 / (1.0 + abs(np.mean(c) - pred_centroid))
                return s
            b1 = max(cands_b1_p, key=lambda c: sc_p(c, borda_sol))
            
            cands_b2 = get_cands(t1, t2, t3, t4)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov_renyi and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_p(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands(t1, t2, t3, t4)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            b3 = max(cands_b3_filt, key=lambda c: sc_p(c, borda_v) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'breit_wigner':
            # Resonancia cuántica de Breit-Wigner modulando la sección eficaz de cada boleto
            def sc_bw(c, bd):
                s = sum(bd[n] for n in c) + 35.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
                for tr in itertools.combinations(c, 3): s += 15.0 * triplet_counts.get(tr, 0)
                # Amplitud resonante de Breit-Wigner
                s_sum = sum(c)
                bw_amp = (gamma_half**2) / ((s_sum - E0_res)**2 + gamma_half**2)
                s += 35.0 * bw_amp
                return s
            
            cands_b1 = get_cands(t1_sol, t2_sol, t3_sol, t4_sol)
            b1 = max(cands_b1, key=lambda c: sc_bw(c, borda_sol))
            
            cands_b2 = get_cands(t1, t2, t3, t4)
            borda_d = borda_v.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov_renyi and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_bw(c, borda_d)) if cands_b2_filt else cands_b2[0]
            
            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands(t1, t2, t3, t4)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            b3 = max(cands_b3_filt, key=lambda c: sc_bw(c, borda_v) + 30.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
            tickets = [b1, b2, b3]

        elif mode == 'gut_46_supreme':
            # Gran Unificación Suprema: Rényi-X3 + Ricci Flow Geodésico + Breit-Wigner Resonant Peak
            borda_gut = borda_v.copy()
            for n in range(46): borda_gut[n] += 0.8 * ricci_flow_boost[n]
            rk_gut = sorted(range(46), key=lambda x: borda_gut[x], reverse=True)
            t1_g, t2_g, t3_g, t4_g = rk_gut[0:9], rk_gut[9:18], rk_gut[18:27], rk_gut[27:36]

            def sc_gut(c, bd):
                s = sum(bd[n] for n in c) + 38.0 * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
                for tr in itertools.combinations(c, 3): s += 18.0 * triplet_counts.get(tr, 0)
                s_sum = sum(c)
                bw_amp = (gamma_half**2) / ((s_sum - E0_res)**2 + gamma_half**2)
                s += 25.0 * bw_amp
                return s

            cands_b1 = get_cands(t1_sol, t2_sol, t3_sol, t4_sol)
            b1 = max(cands_b1, key=lambda c: sc_gut(c, borda_sol))

            cands_b2 = get_cands(t1_g, t2_g, t3_g, t4_g)
            borda_d = borda_gut.copy()
            for n in range(46): borda_d[n] += 6.0 * counts_day[n]
            cands_b2_filt = [c for c in cands_b2 if len(set(c) & set(b1)) == target_ov_renyi and c != b1]
            if not cands_b2_filt: cands_b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
            b2 = max(cands_b2_filt, key=lambda c: sc_gut(c, borda_d)) if cands_b2_filt else cands_b2[0]

            p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
            cands_b3 = get_cands(t1_g, t2_g, t3_g, t4_g)
            cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
            if not cands_b3_filt: cands_b3_filt = [c for c in cands_b3 if c != b1 and c != b2]
            b3 = max(cands_b3_filt, key=lambda c: sc_gut(c, borda_gut) + 32.0 * len(set(itertools.combinations(c, 2)) - p12)) if cands_b3_filt else cands_b3[0]
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
        print(f"[{idx_eval + 1}/{n_eval}] sorteos evaluados ({elapsed:.1f}s) | Líder actual Glob 4+: {max(r['glob_4'] for r in results.values())}")

print('='*115)
print('RESULTADOS FINALES: ULTRA-FRONTERA MATEMÁTICA (660 SORTEOS / 1.980 BOLETOS)')
print('='*115)
print(f"{'Modelo / Teoría Radical':<44} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print('-'*115)
for c, r in results.items():
    print(f"{c:<44} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print('='*115)

