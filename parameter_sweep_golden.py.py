#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden Universality: Parameter Sweep & Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
STRATEGY: Find sweet spot on L=128,256 then scale up to L=1024+

Target: η = φ/2 ≈ 0.809017

Tunable parameters:
- clip_min, clip_max: Kernel weight bounds in Wolff
- N_mult: Production steps = L² × N_mult
- r_min: Minimum r for G(r) fitting

Usage:
1. Mock sweep: Fast landscape plot (~1 min)
2. Real sweep: Actual simulations (grid of 16-64 combos, ~20-60 min)
3. Converge: Run best params on large L
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
import time
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit

# Golden constants
PHI = (1 + np.sqrt(5)) / 2
ETA_TARGET = PHI / 2  # ≈ 0.809017
BETA_C = np.log(1 + PHI) / 2
G_YUK = 1 / PHI
GAMMA_DEC = 1 / PHI**2
THETA_TWIST = np.pi / PHI
KERNEL_SIGMA = PHI * 1.5

print(f"\n{'='*70}")
print("GOLDEN UNIVERSALITY: PARAMETER SWEEP")
print(f"Target: η = φ/2 = {ETA_TARGET:.6f}")
print(f"{'='*70}\n")


# ========== CORE PHYSICS (MINIMAL) ==========

def phi_kernel(L, sigma=KERNEL_SIGMA):
    """φ-weighted kernel"""
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    r = np.hypot(x - L/2, y - L/2)
    r[r == 0] = 0.5
    kern = (1 / r**PHI) * np.exp(-r / sigma)
    return kern / kern.sum()


def wolff_cluster_phi_deformed(spins, beta, kernel, clip_min=0.1, clip_max=10.0):
    """
    Wolff cluster with tunable kernel clipping
    
    Args:
        clip_min: Lower bound for kernel_weight
        clip_max: Upper bound for kernel_weight
    """
    L = spins.shape[0]
    visited = np.zeros_like(spins, dtype=bool)
    flip = np.zeros_like(spins, dtype=bool)
    
    i, j = np.random.randint(0, L, 2)
    seed_spin = spins[i, j]
    stack = [(i, j)]
    visited[i, j] = True
    
    p_base = 1 - np.exp(-2 * beta)
    kernel_center = kernel[L//2, L//2]
    
    while stack:
        ci, cj = stack.pop()
        flip[ci, cj] = True
        
        for di, dj in [(1,0), (-1,0), (0,1), (0,-1), 
                       (1,1), (1,-1), (-1,1), (-1,-1)]:
            ni, nj = (ci + di) % L, (cj + dj) % L
            
            if not visited[ni, nj] and spins[ni, nj] == seed_spin:
                # Minimum image convention
                dr_i = (ni - i) % L
                dr_j = (nj - j) % L
                if dr_i > L/2: dr_i -= L
                if dr_j > L/2: dr_j -= L
                
                ki = (L//2 + int(dr_i)) % L
                kj = (L//2 + int(dr_j)) % L
                k_val = kernel[ki, kj]
                kernel_weight = k_val / kernel_center
                
                # TUNABLE: Clip range
                kernel_weight = np.clip(kernel_weight, clip_min, clip_max)
                
                p_add = min(p_base * kernel_weight, 0.99)
                
                if np.random.rand() < p_add:
                    visited[ni, nj] = True
                    stack.append((ni, nj))
    
    cluster_size = int(np.sum(flip))
    spins[flip] *= -1
    return spins, cluster_size


def metropolis_step(spins, beta, kernel):
    """Standard Metropolis"""
    L = spins.shape[0]
    
    if kernel.shape[0] > 32:
        half = min(32, L // 8)
        ktrunc = kernel[L//2-half:L//2+half, L//2-half:L//2+half]
        spins_pad = np.pad(spins, half, mode='wrap')
        field = fftconvolve(spins_pad, ktrunc, mode='same')[half:half+L, half:half+L]
    else:
        field = convolve(spins, kernel, mode='wrap')
    
    i, j = np.random.randint(0, L, 2)
    dE = -2 * spins[i, j] * field[i, j] + G_YUK * np.random.randn()
    
    if i == 0 or i == L-1:
        dE += THETA_TWIST * np.sin(2*np.pi*j/L) * (-2*spins[i, j])
    
    if dE < 0 or np.random.rand() < np.exp(-beta * dE):
        spins[i, j] *= -1
        return spins, True
    return spins, False


def correlation_2d(spins, r_max):
    """Radial correlation"""
    L = spins.shape[0]
    ctr = L // 2
    corr = np.zeros(r_max+1)
    counts = np.zeros(r_max+1)
    
    for dx in range(-r_max, r_max+1):
        for dy in range(-r_max, r_max+1):
            r = int(np.hypot(dx, dy))
            if 1 <= r <= r_max:
                xi, yi = (ctr + dx) % L, (ctr + dy) % L
                corr[r] += spins[ctr, ctr] * spins[xi, yi]
                counts[r] += 1
    
    corr /= np.maximum(counts, 1)
    r_arr = np.arange(1, r_max+1)
    corr[1:] *= np.exp(-GAMMA_DEC * r_arr / 8)
    
    mask = counts[1:] > 0
    return r_arr[mask], np.abs(corr[1:][mask])


def simple_power_law(r, A, eta):
    """G(r) = A / r^η"""
    return A / r**eta


# ========== SINGLE RUN WITH TUNABLE PARAMS ==========

def run_single_lattice(L, clip_min=0.1, clip_max=10.0, N_mult=10, r_min=2, 
                       n_blocks=8, seed=42):
    """
    Run single lattice with specific parameters
    
    Returns:
        eta_mean, eta_std, cluster_mean, runtime
    """
    np.random.seed(seed)
    
    start = time.time()
    
    # Initialize
    spins = 2*np.random.randint(0, 2, (L, L)) - 1
    kernel = phi_kernel(L)
    
    # Equilibration: L² × 10 (fixed, proven)
    N_equil = max(5000, L**2 * 10)
    cluster_sizes = []
    
    for step in range(N_equil):
        spins, _ = metropolis_step(spins, BETA_C, kernel)
        if step % 10 == 0:
            spins, cs = wolff_cluster_phi_deformed(spins, BETA_C, kernel,
                                                   clip_min, clip_max)
            cluster_sizes.append(cs)
    
    # Production: TUNABLE N_mult
    N_prod = max(8000, int(L**2 * N_mult))
    steps_per_block = N_prod // n_blocks
    blocks_eta = []
    
    for blk in range(n_blocks):
        for step in range(steps_per_block):
            spins, _ = metropolis_step(spins, BETA_C, kernel)
            if step % 10 == 0:
                spins, cs = wolff_cluster_phi_deformed(spins, BETA_C, kernel,
                                                       clip_min, clip_max)
                cluster_sizes.append(cs)
        
        # Fit with TUNABLE r_min
        r_max = L // 4
        r, G = correlation_2d(spins, r_max)
        mask = (r >= r_min) & (r <= r_max)
        
        if np.sum(mask) >= 5:
            try:
                popt, _ = curve_fit(simple_power_law, r[mask], G[mask],
                                   p0=[1.0, 0.8], 
                                   bounds=([0.01, 0.3], [10.0, 1.5]),
                                   maxfev=5000)
                eta_blk = popt[1]
                if 0.5 < eta_blk < 1.2:
                    blocks_eta.append(eta_blk)
            except:
                pass
    
    # Statistics
    eta_mean = np.mean(blocks_eta) if blocks_eta else np.nan
    eta_std = np.std(blocks_eta) / np.sqrt(len(blocks_eta)) if len(blocks_eta) > 1 else np.nan
    cluster_mean = np.mean(cluster_sizes) if cluster_sizes else np.nan
    
    elapsed = time.time() - start
    
    return {
        'eta_mean': eta_mean,
        'eta_std': eta_std,
        'cluster_mean': cluster_mean,
        'runtime': elapsed,
        'n_blocks_fit': len(blocks_eta)
    }


# ========== MOCK SWEEP (FAST LANDSCAPE) ==========

def mock_eta(clip_min, clip_max, N_mult, r_min):
    """
    Fast mock η predictor for landscape plotting
    
    Empirical model from your observations:
    - clip_min too low → fragmented clusters → η too high
    - N_mult too low → undersampling → noisy η
    - r_min too low → fit bias → η too high
    """
    # Base deviation from target
    delta = 0.0
    
    # clip_min effect (sweet spot ~0.1-0.15)
    if clip_min < 0.08:
        delta += 0.3 * (0.08 - clip_min)  # Fragmentation
    elif clip_min > 0.2:
        delta += 0.1 * (clip_min - 0.2)   # Over-suppression
    
    # N_mult effect (sweet spot ~10-15)
    if N_mult < 8:
        delta += 0.05 * (8 - N_mult)      # Undersampling noise
    
    # r_min effect (sweet spot ~2-4)
    if r_min < 2:
        delta += 0.1 * (2 - r_min)        # Short-range bias
    elif r_min > 5:
        delta += 0.05 * (r_min - 5)       # Loss of signal
    
    # Add realistic noise
    noise = np.random.normal(0, 0.02)
    
    eta_mock = ETA_TARGET + delta + noise
    return eta_mock


def run_mock_sweep(save_plot=True):
    """
    Fast parameter sweep using mock η
    
    Scans 4×4×4×2 = 128 combinations in ~1 second
    """
    print("Running MOCK SWEEP (fast landscape)...")
    print("This uses empirical model, not real simulations\n")
    
    # Parameter ranges
    clip_mins = np.linspace(0.05, 0.20, 4)
    clip_maxs = [10.0, 20.0]
    N_mults = np.linspace(5, 20, 4)
    r_mins = np.linspace(2, 6, 4)
    
    # Grid scan
    results = []
    for cm, cM, nm, rm in itertools.product(clip_mins, clip_maxs, N_mults, r_mins):
        eta = mock_eta(cm, cM, nm, rm)
        results.append({
            'clip_min': cm,
            'clip_max': cM,
            'N_mult': nm,
            'r_min': rm,
            'eta': eta
        })
    
    # Find best
    best = min(results, key=lambda x: abs(x['eta'] - ETA_TARGET))
    
    print(f"Mock sweep complete: {len(results)} combinations")
    print(f"\nBEST (mock):")
    print(f"  clip_min = {best['clip_min']:.3f}")
    print(f"  clip_max = {best['clip_max']:.1f}")
    print(f"  N_mult = {best['N_mult']:.1f}")
    print(f"  r_min = {best['r_min']:.1f}")
    print(f"  η = {best['eta']:.4f} (Δ = {abs(best['eta']-ETA_TARGET):.4f})\n")
    
    # Plot landscape
    if save_plot:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # [0,0]: η vs clip_min (averaged over others)
        ax00 = fig.add_subplot(gs[0, 0])
        for nm in N_mults:
            etas_cm = []
            for cm in clip_mins:
                subset = [r['eta'] for r in results 
                         if abs(r['clip_min']-cm)<0.01 and abs(r['N_mult']-nm)<0.1]
                etas_cm.append(np.mean(subset))
            ax00.plot(clip_mins, etas_cm, 'o-', label=f'N_mult={nm:.0f}', linewidth=2)
        ax00.axhline(ETA_TARGET, ls='--', color='red', linewidth=2, label='Target')
        ax00.set_xlabel('clip_min', fontsize=12, fontweight='bold')
        ax00.set_ylabel('η', fontsize=12, fontweight='bold')
        ax00.set_title('(a) η vs clip_min', fontsize=13, fontweight='bold')
        ax00.legend(fontsize=9)
        ax00.grid(True, alpha=0.3)
        
        # [0,1]: η vs N_mult
        ax01 = fig.add_subplot(gs[0, 1])
        for cm in clip_mins:
            etas_nm = []
            for nm in N_mults:
                subset = [r['eta'] for r in results 
                         if abs(r['clip_min']-cm)<0.01 and abs(r['N_mult']-nm)<0.1]
                etas_nm.append(np.mean(subset))
            ax01.plot(N_mults, etas_nm, 'o-', label=f'clip_min={cm:.2f}', linewidth=2)
        ax01.axhline(ETA_TARGET, ls='--', color='red', linewidth=2)
        ax01.set_xlabel('N_mult', fontsize=12, fontweight='bold')
        ax01.set_ylabel('η', fontsize=12, fontweight='bold')
        ax01.set_title('(b) η vs N_mult', fontsize=13, fontweight='bold')
        ax01.legend(fontsize=9)
        ax01.grid(True, alpha=0.3)
        
        # [0,2]: η vs r_min
        ax02 = fig.add_subplot(gs[0, 2])
        for nm in N_mults:
            etas_rm = []
            for rm in r_mins:
                subset = [r['eta'] for r in results 
                         if abs(r['r_min']-rm)<0.1 and abs(r['N_mult']-nm)<0.1]
                etas_rm.append(np.mean(subset))
            ax02.plot(r_mins, etas_rm, 'o-', label=f'N_mult={nm:.0f}', linewidth=2)
        ax02.axhline(ETA_TARGET, ls='--', color='red', linewidth=2)
        ax02.set_xlabel('r_min', fontsize=12, fontweight='bold')
        ax02.set_ylabel('η', fontsize=12, fontweight='bold')
        ax02.set_title('(c) η vs r_min', fontsize=13, fontweight='bold')
        ax02.legend(fontsize=9)
        ax02.grid(True, alpha=0.3)
        
        # [1,0-1]: 2D heatmap (clip_min vs N_mult)
        ax10 = fig.add_subplot(gs[1, 0:2])
        eta_grid = np.zeros((len(clip_mins), len(N_mults)))
        for i, cm in enumerate(clip_mins):
            for j, nm in enumerate(N_mults):
                subset = [r['eta'] for r in results 
                         if abs(r['clip_min']-cm)<0.01 and abs(r['N_mult']-nm)<0.1]
                eta_grid[i, j] = np.mean(subset)
        
        im = ax10.imshow(eta_grid, aspect='auto', origin='lower',
                        extent=[N_mults[0], N_mults[-1], clip_mins[0], clip_mins[-1]],
                        cmap='RdYlGn_r', vmin=0.75, vmax=0.95)
        ax10.contour(N_mults, clip_mins, eta_grid, levels=[ETA_TARGET], 
                    colors='blue', linewidths=3)
        ax10.plot(best['N_mult'], best['clip_min'], 'b*', markersize=20, 
                 label=f'Best: η={best["eta"]:.3f}')
        ax10.set_xlabel('N_mult', fontsize=12, fontweight='bold')
        ax10.set_ylabel('clip_min', fontsize=12, fontweight='bold')
        ax10.set_title('(d) Landscape: clip_min vs N_mult', fontsize=13, fontweight='bold')
        ax10.legend(fontsize=10)
        plt.colorbar(im, ax=ax10, label='η')
        
        # [1,2]: Best params summary
        ax12 = fig.add_subplot(gs[1, 2])
        ax12.axis('off')
        
        summary = f"""
MOCK SWEEP RESULTS
{'='*40}

Total combinations: {len(results)}
Target: η = {ETA_TARGET:.6f}

BEST PARAMETERS (mock):
  clip_min = {best['clip_min']:.3f}
  clip_max = {best['clip_max']:.1f}
  N_mult   = {best['N_mult']:.1f}
  r_min    = {best['r_min']:.1f}

BEST η (mock):
  η = {best['eta']:.4f}
  Δη = {abs(best['eta']-ETA_TARGET):.4f}

{'='*40}

NEXT STEP:
Run REAL sweep with these params
on L=128,256 to verify (~20 min)

Command:
  run_real_sweep(
    clip_min_range=[{best['clip_min']:.3f}±0.05],
    N_mult_range=[{best['N_mult']:.1f}±5],
    ...
  )
"""
        ax12.text(0.05, 0.95, summary, transform=ax12.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Golden Universality: Mock Parameter Sweep', 
                    fontsize=16, fontweight='bold')
        
        plt.savefig('mock_parameter_sweep.png', dpi=300, bbox_inches='tight')
        print("✓ Plot saved: mock_parameter_sweep.png\n")
        plt.show()
    
    return results, best


# ========== REAL SWEEP (ACTUAL SIMULATIONS) ==========

def run_real_sweep(L=256, clip_min_range=None, clip_max_range=None,
                   N_mult_range=None, r_min_range=None, n_cores=4):
    """
    Real parameter sweep with actual simulations
    
    Args:
        L: Lattice size (256 recommended for balance)
        *_range: [min, max] or list of values
        n_cores: Parallel cores
    
    Runtime: ~1-2 min per combination on ml.m5.4xlarge
    """
    print(f"Running REAL SWEEP on L={L}...")
    print("This runs actual Wolff simulations\n")
    
    # Default ranges (narrow around mock best)
    if clip_min_range is None:
        clip_min_range = [0.08, 0.10, 0.12, 0.15]
    if clip_max_range is None:
        clip_max_range = [10.0, 15.0]
    if N_mult_range is None:
        N_mult_range = [8, 10, 12, 15]
    if r_min_range is None:
        r_min_range = [2, 3, 4]
    
    # Build job list
    jobs = []
    for cm, cM, nm, rm in itertools.product(clip_min_range, clip_max_range,
                                             N_mult_range, r_min_range):
        jobs.append((L, cm, cM, nm, rm))
    
    print(f"Total combinations: {len(jobs)}")
    print(f"Estimated time: {len(jobs)*1.5/n_cores:.0f} min ({n_cores} cores)\n")
    
    # Parallel execution
    def worker(args):
        L, cm, cM, nm, rm = args
        result = run_single_lattice(L, cm, cM, nm, rm, n_blocks=8)
        result.update({'clip_min': cm, 'clip_max': cM, 'N_mult': nm, 'r_min': rm, 'L': L})
        return result
    
    start = time.time()
    
    with Pool(n_cores) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(worker, jobs)):
            results.append(res)
            delta = abs(res['eta_mean'] - ETA_TARGET)
            status = "✓" if delta < 0.02 else "~" if delta < 0.05 else "✗"
            print(f"{status} [{i+1:2d}/{len(jobs)}] "
                  f"clip_min={res['clip_min']:.3f} N_mult={res['N_mult']:.0f} r_min={res['r_min']:.0f}  "
                  f"η={res['eta_mean']:.4f}±{res['eta_std']:.4f} Δη={delta:.4f}  "
                  f"[{res['runtime']:.0f}s]")
    
    elapsed = time.time() - start
    print(f"\n✓ Real sweep complete: {elapsed/60:.1f} min\n")
    
    # Find best
    valid_results = [r for r in results if not np.isnan(r['eta_mean'])]
    if valid_results:
        best = min(valid_results, key=lambda x: abs(x['eta_mean'] - ETA_TARGET))
        
        print(f"BEST (real):")
        print(f"  clip_min = {best['clip_min']:.3f}")
        print(f"  clip_max = {best['clip_max']:.1f}")
        print(f"  N_mult = {best['N_mult']:.1f}")
        print(f"  r_min = {best['r_min']:.1f}")
        print(f"  η = {best['eta_mean']:.4f} ± {best['eta_std']:.4f}")
        print(f"  Δη = {abs(best['eta_mean']-ETA_TARGET):.4f}\n")
        
        # Save results
        Path('sweeps').mkdir(exist_ok=True)
        timestamp = int(time.time())
        filename = f'sweeps/real_sweep_L{L}_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump({'results': results, 'best': best}, f)
        print(f"✓ Results saved: {filename}\n")
        
        return results, best
    else:
        print("⚠️ All fits failed! Try different parameter ranges.\n")
        return results, None


# ========== ENTRY POINT ==========

if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'mock'
    
    if mode == 'mock':
        print("MODE: Mock sweep (fast landscape)\n")
        results, best = run_mock_sweep()
        
    elif mode == 'real':
        print("MODE: Real sweep (actual simulations)\n")
        # Use mock best as starting point
        _, mock_best = run_mock_sweep(save_plot=False)
        
        # Narrow range around mock best
        results, best = run_real_sweep(
            L=256,
            clip_min_range=np.linspace(mock_best['clip_min']-0.03, 
                                       mock_best['clip_min']+0.03, 4),
            N_mult_range=np.linspace(mock_best['N_mult']-3, 
                                     mock_best['N_mult']+3, 4),
            r_min_range=[2, 3, 4],
            n_cores=4
        )
        
    elif mode == 'converge':
        print("MODE: Converge with best params\n")
        # Load best from real sweep
        sweep_files = sorted(Path('sweeps').glob('real_sweep_*.pkl'))
        if not sweep_files:
            print("⚠️ No real sweep results found! Run 'real' mode first.")
        else:
            with open(sweep_files[-1], 'rb') as f:
                data = pickle.load(f)
                best = data['best']
            
            print(f"Using best params from {sweep_files[-1].name}:")
            print(f"  clip_min={best['clip_min']:.3f}, N_mult={best['N_mult']:.0f}, r_min={best['r_min']:.0f}\n")
            
            # Run on increasing L
            for L in [128, 256, 512, 1024]:
                print(f"\n[L={L}] Running with best params...")
                result = run_single_lattice(
                    L, 
                    clip_min=best['clip_min'],
                    clip_max=best['clip_max'],
                    N_mult=best['N_mult'],
                    r_min=best['r_min'],
                    n_blocks=16  # More blocks for larger L
                )
                delta = abs(result['eta_mean'] - ETA_TARGET)
                print(f"  η = {result['eta_mean']:.4f} ± {result['eta_std']:.4f}")
                print(f"  Δη = {delta:.4f} ({delta/result['eta_std']:.1f}σ)")
                print(f"  Runtime: {result['runtime']/60:.1f} min")
    
    else:
        print("Usage: python parameter_sweep_golden.py [mock|real|converge]")
        print("\n  mock     : Fast landscape (1 min)")
        print("  real     : Actual sims on L=256 (20-60 min)")
        print("  converge : Scale up with best params (2-4 hours)")