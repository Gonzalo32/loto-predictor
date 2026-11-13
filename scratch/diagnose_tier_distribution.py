import os, sys, itertools, time
from collections import defaultdict, Counter
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)

from generador_cobertura_ml import load_data, compute_delta_delay_momentum_scores
from scratch.backtest_600_sorteos_produccion import compute_fast_attention_mlp_scores

draws = load_data(os.path.join(WORKSPACE_DIR, 'historico_quini_completo.csv'))
trad_draws = [d for d in draws if d['modalidad'] == 'Tradicional']
n_eval = 660
eval_draws = trad_draws[-n_eval:]

tier_counts_trad = Counter()
tier_counts_glob = Counter()
t1_counts_trad = Counter()
t1_counts_glob = Counter()

for idx_eval, target_draw in enumerate(eval_draws):
    target_s = target_draw['sorteo']
    train_draws = [d for d in draws if d['sorteo'] < target_s]
    train_trad = [d for d in train_draws if d['modalidad'] == 'Tradicional']
    
    all_target_draws = [d for d in draws if d['sorteo'] == target_s]
    real_by_mod = {d['modalidad']: set(d['numbers']) for d in all_target_draws}
    real_trad = real_by_mod.get('Tradicional', set())
    
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

    rk_b = sorted(range(46), key=lambda x: borda_pure[x], reverse=True)
    t1 = set(rk_b[0:9])
    t2 = set(rk_b[9:18])
    t3 = set(rk_b[18:27])
    t4 = set(rk_b[27:36])
    t5 = set(rk_b[36:46])
    
    def get_tier_tuple(nums):
        c1 = len(nums & t1)
        c2 = len(nums & t2)
        c3 = len(nums & t3)
        c4 = len(nums & t4)
        c5 = len(nums & t5)
        return (c1, c2, c3, c4, c5)
    
    if real_trad:
        tt = get_tier_tuple(real_trad)
        tier_counts_trad[tt] += 1
        t1_counts_trad[tt[0]] += 1
        
    for mod, nums in real_by_mod.items():
        tt_g = get_tier_tuple(nums)
        tier_counts_glob[tt_g] += 1
        t1_counts_glob[tt_g[0]] += 1

print("="*80)
print(f"ANÁLISIS DE DISTRIBUCIÓN REAL DE TIERS ({n_eval} SORTEOS)")
print("="*80)
print("\n¿Cuántas bolillas caen en TIER 1 (Top 9 más probables)?")
print(f"Tradicional:")
for k in range(7):
    print(f"  {k} bolillas en Tier 1: {t1_counts_trad[k]:3d} veces ({t1_counts_trad[k]/n_eval*100:.1f}%)")

print(f"\nGlobal (todas las modalidades):")
tot_draws_g = sum(t1_counts_glob.values())
for k in range(7):
    print(f"  {k} bolillas en Tier 1: {t1_counts_glob[k]:4d} veces ({t1_counts_glob[k]/tot_draws_g*100:.1f}%)")

print("\nTop 10 Configuraciones de Tiers más frecuentes (T1, T2, T3, T4, T5):")
for tt, cnt in tier_counts_trad.most_common(10):
    print(f"  Config {tt}: {cnt:3d} veces ({cnt/n_eval*100:.1f}%)")

