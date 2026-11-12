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

# ---------------------------------------------------------------------------
# Precomputos de Teoría de Números y Álgebra Abstracta sobre F_47
# ---------------------------------------------------------------------------
# 47 es primo. 5 es raíz primitiva módulo 47 (genera todo el grupo cíclico F_47*).
PRIMITIVE_ROOT = 5
DLOG_47 = {}
val = 1
for exp in range(46):
    DLOG_47[val] = exp
    val = (val * PRIMITIVE_ROOT) % 47

F47_QR = set(pow(x, 2, 47) for x in range(1, 47)) # 23 elementos

# ---------------------------------------------------------------------------
# Configuraciones Radicales a Probar:
# ---------------------------------------------------------------------------
# 0. Galois F_47 Champion (Actual mejor: 18 Glob 4+, 6 Trad 4+)
# 1. Armónicos Espectrales de Dirichlet (Transformada de Fourier sobre F_47*)
# 2. Hipergrafo Tensor de 3 Cuerpos (Triplet Hyperedge Centrality)
# 3. Propagación de Solitones KdV en el Toro Z_46 (Soliton Wavefront)
# 4. Entrelazamiento Cuántico Multi-Modal (EPR Cross-Modal Bridge)
# 5. La Ecuación Maestra (Súper-Ensamble Galois + Dirichlet + Hipergrafo)

configs = {
    '0. Galois F_47 Champion (18 Glob 4+)': {'mode': 'galois_champion'},
    '1. Armónicos de Dirichlet (Fourier F_47)': {'mode': 'dirichlet'},
    '2. Hipergrafo de 3 Cuerpos (Triplets)':    {'mode': 'hypergraph'},
    '3. Solitones KdV en Z_46':                 {'mode': 'soliton'},
    '4. Entrelazamiento Multi-Modal (EPR)':      {'mode': 'epr_entanglement'},
    '5. La Ecuación Maestra (Galois+Dirichlet)': {'mode': 'master_equation'}
}

results = {c: {'trad_4': 0, 'glob_4': 0, 'trad_3': 0, 'glob_3': 0, 'pe_5': 0, 'secos': 0} for c in configs}

print("="*115)
print(f"EXPERIMENTO DE RUPTURA DE LÍMITES: 5 TEORÍAS DE FRONTERA ({n_eval} SORTEOS / 1.980 BOLETOS)")
print("Probando Armónicos de Dirichlet en F_47, Hipergrafos de 3 Cuerpos, Solitones KdV y Entrelazamiento EPR...")
print("="*115)

for target_draw in eval_draws:
    target_s = target_draw['sorteo']
    target_date_str = target_draw['fecha']
    target_weekday = datetime.strptime(target_date_str, '%Y-%m-%d').weekday()
    
    train_draws = [d for d in draws if d['sorteo'] < target_s]
    train_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    train_seg  = [d for d in train_draws if d['modalidad'] == 'Segunda']
    train_rev  = [d for d in train_draws if d['modalidad'] == 'Revancha']
    
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
    t1 = ranked[0:9]
    t2 = ranked[9:18]
    t3 = ranked[18:27]
    t4 = ranked[27:36]

    # -----------------------------------------------------------------------
    # 1. Armónicos de Dirichlet sobre F_47* (Transformada Espectral de Galois)
    # -----------------------------------------------------------------------
    # Mapeamos cada sorteo a sus logaritmos discretos mod 47 (toro Z_46)
    # y calculamos la densidad de energía espectral para cada armónico k en {1..45}
    dlog_draws = []
    for d in train_trad[-30:]:
        dlog_draws.extend([DLOG_47[num + 1] for num in d['numbers']])
    
    # Coeficientes de Fourier-Dirichlet: c_k = sum(exp(-2pi*i * k * dlog / 46))
    dirichlet_energies = np.zeros(46)
    if dlog_draws:
        for k in range(1, 24): # simetría hermitiana
            angles = 2.0 * np.pi * k * np.array(dlog_draws) / 46.0
            coeff_re = np.sum(np.cos(angles))
            coeff_im = np.sum(np.sin(angles))
            energy = coeff_re**2 + coeff_im**2
            # Reinyectar energía a los números con esa fase
            for n in range(46):
                log_val = DLOG_47[n + 1]
                dirichlet_energies[n] += energy * math.cos(2.0 * math.pi * k * log_val / 46.0)
        dirichlet_energies = (dirichlet_energies - dirichlet_energies.min()) / (dirichlet_energies.max() - dirichlet_energies.min() + 1e-6)

    # -----------------------------------------------------------------------
    # 2. Hipergrafo de 3 Cuerpos (Triplet Hyperedge Tensor Centrality)
    # -----------------------------------------------------------------------
    # Construimos tensores de tripletas frecuentes en los ultimos 40 sorteos
    triplet_counts = Counter()
    for d in train_trad[-40:]:
        for tr in itertools.combinations(sorted(d['numbers']), 3):
            triplet_counts[tr] += 1
    # Centralidad de hipergrafo para cada número: suma de pesos de hiper-aristas
    hyper_centrality = np.zeros(46)
    for (u, v, w_num), cnt in triplet_counts.items():
        if cnt >= 2:
            hyper_centrality[u] += cnt * 2.0
            hyper_centrality[v] += cnt * 2.0
            hyper_centrality[w_num] += cnt * 2.0

    # -----------------------------------------------------------------------
    # 3. Propagación de Solitones KdV en el Toro Cíclico Z_46
    # -----------------------------------------------------------------------
    # Representamos las 46 bolillas en un círculo. Buscamos el frente de onda solitónico
    # que se propaga por inercia sin dispersarse: u(x, t) = A * sech^2((x - v*t)/W)
    dens_circ = np.zeros(46)
    for d in train_trad[-15:]:
        for n in d['numbers']:
            dens_circ[n] += 1.0
    # Convolución circular con kernel sech^2 de ancho W=3
    kernel_sech = np.array([1.0 / (math.cosh(dx/1.5)**2) for dx in range(-4, 5)])
    kernel_sech /= kernel_sech.sum()
    soliton_field = np.zeros(46)
    for n in range(46):
        s_val = 0.0
        for k_idx, dx in enumerate(range(-4, 5)):
            circ_idx = (n + dx) % 46
            s_val += dens_circ[circ_idx] * kernel_sech[k_idx]
        soliton_field[n] = s_val

    # -----------------------------------------------------------------------
    # 4. Entrelazamiento Cuántico Multi-Modal (EPR Cross-Modal Bridge)
    # -----------------------------------------------------------------------
    # Si la bolilla n salió en Tradicional, ¿con qué probabilidad se entrelaza
    # positivamente en Segunda o Revancha en el mismo sorteo?
    cross_epr_score = np.zeros(46)
    for d_prev in train_draws[-60:]:
        # Detectar números que aparecieron en 2 modalidades del mismo sorteo
        pass
    # Bonus a números que muestran alta correlación cruzada entre modalidades
    double_hits = Counter()
    for s_idx in range(target_s - 30, target_s):
        s_draws = [d for d in train_draws if d['sorteo'] == s_idx]
        if len(s_draws) >= 2:
            all_s_nums = Counter(n for d in s_draws for n in d['numbers'])
            for n, cnt in all_s_nums.items():
                if cnt >= 2: double_hits[n] += 1
    for n in range(46):
        cross_epr_score[n] = double_hits[n] * 15.0

    # -----------------------------------------------------------------------
    # Helper Generator para evaluar cada teoría bajo la arquitectura de 3 boletos
    # -----------------------------------------------------------------------
    def run_tickets_generation(b_dict, extra_cand_filter=None, triplet_scoring=False, aff_w=35.0):
        cands = []
        for c_t1 in itertools.combinations(t1[:6], 3):
            for n2 in t2[:3]:
                for n3 in t3[:2]:
                    for n4 in t4[:2]:
                        cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                        if not es_ticket_valido(cand, min_distinct_decades=3): continue
                        if extra_cand_filter and not extra_cand_filter(cand): continue
                        cands.append(cand)
        if len(cands) < 3:
            cands = list(itertools.combinations(t1[:6] + t2[:3], 6))[:10]
            
        def sc(c, bd):
            s = sum(bd[n] for n in c)
            if aff_w > 0:
                s += aff_w * sum(cooccur_mat[u, v] for u, v in itertools.combinations(c, 2))
            if triplet_scoring:
                # Add 3-body hyperedge lift
                for tr in itertools.combinations(c, 3):
                    s += 15.0 * triplet_counts.get(tr, 0)
            return s
            
        cands_sorted = sorted(cands, key=lambda c: sc(c, b_dict), reverse=True)
        b1 = cands_sorted[0]
        
        b_d = b_dict.copy()
        for n in range(46): b_d[n] += 6.0 * counts_day[n]
        cands_b2 = sorted(cands, key=lambda c: sc(c, b_d), reverse=True)
        b2_filt = [c for c in cands_b2 if 2 <= len(set(c) & set(b1)) <= 4 and c != b1]
        b2 = b2_filt[0] if b2_filt else cands_b2[1]
        
        p12 = set(itertools.combinations(b1, 2)) | set(itertools.combinations(b2, 2))
        b3_filt = [c for c in cands_sorted if c != b1 and c != b2 and 2 <= len(set(c) & set(b1)) <= 4 and 2 <= len(set(c) & set(b2)) <= 4]
        b3 = max(b3_filt, key=lambda c: sc(c, b_dict) + 25.0 * len(set(itertools.combinations(c, 2)) - p12)) if b3_filt else cands_sorted[2]
        return [b1, b2, b3]

    # Evaluamos cada una de las 6 teorías
    for c_name, c_cfg in configs.items():
        mode = c_cfg['mode']
        
        if mode == 'galois_champion':
            def qr_filt(cand):
                qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                return 2 <= qr_c <= 4
            tickets = run_tickets_generation(borda_v, extra_cand_filter=qr_filt)
            
        elif mode == 'dirichlet':
            borda_dir = borda_v.copy()
            for n in range(46): borda_dir[n] += 40.0 * dirichlet_energies[n]
            def qr_filt(cand):
                qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                return 2 <= qr_c <= 4
            tickets = run_tickets_generation(borda_dir, extra_cand_filter=qr_filt)
            
        elif mode == 'hypergraph':
            # Usa tensor de 3 cuerpos explícito en la función de scoring
            def qr_filt(cand):
                qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                return 2 <= qr_c <= 4
            tickets = run_tickets_generation(borda_v, extra_cand_filter=qr_filt, triplet_scoring=True)
            
        elif mode == 'soliton':
            borda_sol = borda_v.copy()
            for n in range(46): borda_sol[n] += 10.0 * soliton_field[n]
            def qr_filt(cand):
                qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                return 2 <= qr_c <= 4
            tickets = run_tickets_generation(borda_sol, extra_cand_filter=qr_filt)
            
        elif mode == 'epr_entanglement':
            borda_epr = borda_v.copy()
            for n in range(46): borda_epr[n] += cross_epr_score[n]
            def qr_filt(cand):
                qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                return 2 <= qr_c <= 4
            tickets = run_tickets_generation(borda_epr, extra_cand_filter=qr_filt)
            
        elif mode == 'master_equation':
            # La Ecuación Maestra: Galois F_47 + Dirichlet + Hipergrafo de 3 Cuerpos
            borda_master = borda_v.copy()
            for n in range(46):
                borda_master[n] += 25.0 * dirichlet_energies[n] + 0.5 * hyper_centrality[n]
            def qr_filt(cand):
                qr_c = sum(1 for x in cand if (x+1) in F47_QR)
                return 2 <= qr_c <= 4
            tickets = run_tickets_generation(borda_master, extra_cand_filter=qr_filt, triplet_scoring=True)

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

print('='*115)
print('RESULTADOS: RUPTURA DE LÍMITES MATEMÁTICOS (660 SORTEOS / 1.980 BOLETOS)')
print('='*115)
print(f"{'Modelo / Teoría Radical':<42} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print('-'*115)
for c, r in results.items():
    print(f"{c:<42} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print('='*115)

