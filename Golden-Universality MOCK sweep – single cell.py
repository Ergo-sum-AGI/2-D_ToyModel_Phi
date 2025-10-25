# ==========================================================
# Golden-Universality MOCK sweep – single cell, no L needed
# ==========================================================
import numpy as np, matplotlib.pyplot as plt, itertools

PHI = (1 + np.sqrt(5)) / 2
ETA_TARGET = PHI / 2          # ≈ 0.809017

def mock_eta(clip_min, clip_max, N_mult, r_min):
    delta = 0.0
    if clip_min < 0.08:       delta += 0.3 * (0.08 - clip_min)
    elif clip_min > 0.20:     delta += 0.1 * (clip_min - 0.20)
    if N_mult < 8:            delta += 0.05 * (8 - N_mult)
    if r_min < 2:             delta += 0.1 * (2 - r_min)
    elif r_min > 5:           delta += 0.05 * (r_min - 5)
    return ETA_TARGET + delta + np.random.normal(0, 0.02)

# ---- grid scan ----
clip_mins = np.linspace(0.05, 0.20, 4)
clip_maxs = [10.0, 20.0]
N_mults   = np.linspace(5, 20, 4)
r_mins    = np.linspace(2, 6, 4)

results = []
for cm, cM, nm, rm in itertools.product(clip_mins, clip_maxs, N_mults, r_mins):
    results.append({'clip_min': cm, 'clip_max': cM,
                    'N_mult': nm,   'r_min': rm,
                    'eta': mock_eta(cm, cM, nm, rm)})

best = min(results, key=lambda x: abs(x['eta'] - ETA_TARGET))
print(f"Best (mock): η={best['eta']:.4f}  "
      f"clip_min={best['clip_min']:.3f}  N_mult={best['N_mult']:.1f}  r_min={best['r_min']:.1f}")

# ---- quick plot ----
plt.figure(figsize=(6,4))
eta_grid = np.array([[np.mean([r['eta'] for r in results
                               if abs(r['clip_min']-cm)<1e-3 and abs(r['N_mult']-nm)<1e-1])
                      for nm in N_mults] for cm in clip_mins])
plt.imshow(eta_grid, origin='lower', aspect='auto',
           extent=[N_mults[0], N_mults[-1], clip_mins[0], clip_mins[-1]],
           cmap='RdYlGn_r', vmin=0.75, vmax=0.95)
plt.colorbar(label='η'); plt.contour(N_mults, clip_mins, eta_grid, levels=[ETA_TARGET], colors='b')
plt.plot(best['N_mult'], best['clip_min'], 'b*', ms=15); plt.xlabel('N_mult'); plt.ylabel('clip_min')
plt.title('Mock sweep landscape'); plt.savefig('mock_parameter_sweep.png', dpi=300, bbox_inches='tight'); plt.show()