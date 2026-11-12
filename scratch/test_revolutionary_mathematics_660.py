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

# Precomputar residuos cuadraticos en F_47
F47_QR = set(pow(x, 2, 47) for x in range(1, 47)) # 23 elementos

configs = {
    '0. H4 Champion (Referencia)':       {'mode': 'h4'},
    '1. Geometría de Fisher (Geodésica)':{'mode': 'fisher'},
    '2. Interferencia Cuántica de Fase': {'mode': 'quantum'},
    '3. Topología de Agujeros TDA':      {'mode': 'tda'},
    '4. Escape de Kramers (No-Lineal)':  {'mode': 'kramers'},
    '5. Simetría de Galois F_47':        {'mode': 'galois'},
    '6. Gran Fusión Científica (H4+Q+K)':{'mode': 'grand_fusion'}
}

results = {c: {'trad_4': 0, 'glob_4': 0, 'trad_3': 0, 'glob_3': 0, 'pe_5': 0, 'secos': 0} for c in configs}

print("="*115)
print(f"BENCHMARK CIENTÍFICO AVANZADO: 6 NUEVAS TEORÍAS MATEMÁTICAS ({n_eval} SORTEOS / 1.980 BOLETOS)")
print("Evaluando Geometría de Fisher, Fase Cuántica, TDA, Escape de Kramers y Cuerpos de Galois F_47...")
print("="*115)

for target_draw in eval_draws:
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

    # Base Tiers for H4
    ranked = sorted(range(46), key=lambda x: borda_v[x], reverse=True)
    t1 = ranked[0:9]
    t2 = ranked[9:18]
    t3 = ranked[18:27]
    t4 = ranked[27:36]
    
    # -------------------------------------------------------------
    # 1. Geometría de Fisher (Velocidad Geodésica en el Símplice)
    # -------------------------------------------------------------
    counts_t_prev = Counter(n for d in train_trad[-120:-60] for n in d['numbers'])
    p_prev = np.array([counts_t_prev[n] for n in range(46)], dtype=float) + 1e-4
    p_prev /= p_prev.sum()
    p_curr = p_t + 1e-4
    # Fisher velocity: v_i = (p_curr - p_prev) / sqrt(p_prev)
    v_fisher = (p_curr - p_prev) / np.sqrt(p_prev)
    # Geodesic step: p_geo \propto p_curr * exp(1.5 * v_fisher / sqrt(p_curr))
    step_fisher = np.exp(1.2 * v_fisher / np.sqrt(p_curr))
    borda_fisher = borda_v.copy()
    for n in range(46):
        borda_fisher[n] *= step_fisher[n]
    rk_fisher = sorted(range(46), key=lambda x: borda_fisher[x], reverse=True)
    t1_fisher = rk_fisher[0:9]
    t2_fisher = rk_fisher[9:18]
    t3_fisher = rk_fisher[18:27]
    t4_fisher = rk_fisher[27:36]

    # -------------------------------------------------------------
    # 2. Interferencia Cuántica de Fase (Fases Armónicas & Coherencia)
    # -------------------------------------------------------------
    delays = np.zeros(46)
    for n in range(46):
        d_cnt = 0
        for d in reversed(train_trad):
            if n in d['numbers']: break
            d_cnt += 1
        delays[n] = d_cnt
    mean_cycle = 46.0 / 6.0 # ~7.66 sorteos
    phases = 2.0 * np.pi * (delays / mean_cycle % 1.0)
    # Quantum coherence matrix Q_mat: I(u, v) = 2 * sqrt(p_u * p_v) * cos(phi_u - phi_v)
    q_mat = np.zeros((46, 46))
    for u in range(46):
        for v in range(u+1, 46):
            coherence = np.cos(phases[u] - phases[v])
            q_mat[u, v] = coherence
            q_mat[v, u] = coherence

    # -------------------------------------------------------------
    # 3. Topología de Agujeros TDA (Colapso Gravitacional de Vacíos)
    # -------------------------------------------------------------
    recent_5 = set(n for d in train_trad[-5:] for n in d['numbers'])
    void_scores = np.zeros(46)
    # Find contiguous numbers not hit in 5 draws (topological 1-holes)
    void_len = 0
    void_start = 0
    for n in range(46):
        if n not in recent_5:
            if void_len == 0: void_start = n
            void_len += 1
        else:
            if void_len >= 5:
                # Void collapse: boundary and center numbers get gravitational pull
                for k in range(void_start, void_start + void_len):
                    void_scores[k] = 40.0
            void_len = 0
    borda_tda = borda_v.copy()
    for n in range(46): borda_tda[n] += void_scores[n]
    rk_tda = sorted(range(46), key=lambda x: borda_tda[x], reverse=True)
    t1_tda, t2_tda, t3_tda, t4_tda = rk_tda[0:9], rk_tda[9:18], rk_tda[18:27], rk_tda[27:36]

    # -------------------------------------------------------------
    # 4. Escape de Kramers (Tasa de Tunelamiento No-Lineal para Atrasos)
    # -------------------------------------------------------------
    # Barrier potential V(D) = max(0, 15 - D)^1.5; when D > 15, barrier breaks
    kramers_rates = np.zeros(46)
    for n in range(46):
        D = delays[n]
        if D >= 14:
            # Over the barrier: explosive escape probability
            kramers_rates[n] = 50.0 * math.exp(min(2.0, (D - 14) * 0.25))
        elif D >= 8:
            # Approaching barrier
            kramers_rates[n] = 20.0 * (D / 14.0)
        else:
            kramers_rates[n] = 0.0
    rk_kramers = sorted(range(46), key=lambda x: borda_v[x] + kramers_rates[x], reverse=True)
    t1_kramers, t2_kramers, t3_kramers, t4_kramers = rk_kramers[0:9], rk_kramers[9:18], rk_kramers[18:27], rk_kramers[27:36]

    # Helper function to generate H4-style tickets from given tiers and scoring matrix
    def generate_tickets_from_tiers(tiers_1, tiers_2, tiers_3, tiers_4, b_dict, aff_mat, aff_weight, use_galois=False):
        cands = []
        for c_t1 in itertools.combinations(tiers_1[:6], 3):
            for n2 in tiers_2[:3]:
                for n3 in tiers_3[:2]:
                    for n4 in tiers_4[:2]:
                        cand = tuple(sorted(list(c_t1) + [n2, n3, n4]))
                        if not es_ticket_valido(cand, min_distinct_decades=3): continue
                        if use_galois:
                            # Require balanced quadratic residues in F_47 (between 2 and 4 QR)
                            qr_count = sum(1 for x in cand if (x+1) in F47_QR)
                            if qr_count < 2 or qr_count > 4: continue
                        cands.append(cand)
        if len(cands) < 3:
            cands = list(itertools.combinations(tiers_1[:6] + tiers_2[:3], 6))[:10]
            
        def sc(c, bd):
            s = sum(bd[n] for n in c)
            if aff_weight > 0:
                s += aff_weight * sum(aff_mat[u, v] for u, v in itertools.combinations(c, 2))
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

    for c_name, c_cfg in configs.items():
        mode = c_cfg['mode']
        
        if mode == 'h4':
            tickets = generate_tickets_from_tiers(t1, t2, t3, t4, borda_v, cooccur_mat, 35.0)
        elif mode == 'fisher':
            tickets = generate_tickets_from_tiers(t1_fisher, t2_fisher, t3_fisher, t4_fisher, borda_fisher, cooccur_mat, 35.0)
        elif mode == 'quantum':
            # Use Quantum Coherence Matrix instead of empirical co-occurrence!
            tickets = generate_tickets_from_tiers(t1, t2, t3, t4, borda_v, q_mat, 45.0)
        elif mode == 'tda':
            tickets = generate_tickets_from_tiers(t1_tda, t2_tda, t3_tda, t4_tda, borda_tda, cooccur_mat, 35.0)
        elif mode == 'kramers':
            tickets = generate_tickets_from_tiers(t1_kramers, t2_kramers, t3_kramers, t4_kramers, borda_v, cooccur_mat, 35.0)
        elif mode == 'galois':
            tickets = generate_tickets_from_tiers(t1, t2, t3, t4, borda_v, cooccur_mat, 35.0, use_galois=True)
        elif mode == 'grand_fusion':
            # Fusion: Fisher velocity in ranking + Quantum coherence in pairs + Kramers escape in delay tier
            borda_gf = borda_fisher.copy()
            for n in range(46): borda_gf[n] += 0.5 * kramers_rates[n]
            hybrid_matrix = cooccur_mat + 0.5 * q_mat
            tickets = generate_tickets_from_tiers(t1_fisher, t2_fisher, t3_fisher, t4_kramers, borda_gf, hybrid_matrix, 40.0)

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
print('RESULTADOS COMPARATIVOS: NUEVAS TEORÍAS MATEMÁTICAS (660 SORTEOS / 1.980 BOLETOS)')
print('='*115)
print(f"{'Modelo / Teoría Matemática':<40} | {'Trad 4+':<8} | {'Glob 4+':<8} | {'Trad 3+':<8} | {'Glob 3+':<8} | {'PE 5+':<7} | {'Secos':<8}")
print('-'*115)
for c, r in results.items():
    print(f"{c:<40} | {r['trad_4']:^8} | {r['glob_4']:^8} | {r['trad_3']:^8} | {r['glob_3']:^8} | {r['pe_5']:^7} | {r['secos']:^8}")
print('='*115)

