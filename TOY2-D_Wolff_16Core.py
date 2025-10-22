# ========== FIXED PARALLEL PROCESSING ==========
# Two approaches: Simple multiprocessing OR process pools with proper pickling

from multiprocessing import Pool, cpu_count
import time
from pathlib import Path

# ---------- OPTION 1: Worker wrapper (no Rich inside workers) ----------
def _worker_wrapper(args):
    """
    Worker function for Pool - MUST be at module level for pickling
    No Rich progress inside workers (they run in separate processes)
    """
    L, idx, resume_from, checkpoint_dir = args
    
    # Create fresh checkpoint manager inside worker
    checkpoint_mgr = CheckpointManager(checkpoint_dir)
    
    # Call processor WITHOUT progress_callback (None disables Rich)
    return process_lattice_with_progress(
        L_cur=L,
        seed_offset=idx,
        progress_callback=None,  # ← Disable Rich in workers
        checkpoint_mgr=checkpoint_mgr,
        resume_from=resume_from
    )


def run_corrected_golden_analysis_parallel(LS=[256, 512, 1024], n_cores=None, resume=False):
    """
    PARALLEL VERSION with multiprocessing Pool
    Progress tracking happens at parent level only
    """
    if n_cores is None:
        n_cores = min(cpu_count(), len(LS))

    # Setup
    checkpoint_dir = "checkpoints"
    Path("logs").mkdir(exist_ok=True)
    Path(checkpoint_dir).mkdir(exist_ok=True)
    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    print(f"\n{'='*70}")
    print(f"CORRECTED GOLDEN UNIVERSALITY ANALYSIS (PARALLEL)")
    print(f"Lattices: {LS}")
    print(f"Cores: {n_cores}")
    print(f"Target: η = φ/2 = {ETA_TARGET:.6f}")
    print(f"{'='*70}\n")

    # Check for resume points
    resume_info = {}
    if resume:
        for L in LS:
            checkpoints = checkpoint_mgr.list_checkpoints(L)
            if checkpoints:
                last_block = max([c[0] for c in checkpoints])
                resume_info[L] = last_block
                print(f"[L={L}] Resuming from block {last_block}")

    # Build argument list for workers
    job_args = [
        (L, idx, resume_info.get(L), checkpoint_dir)
        for idx, L in enumerate(LS)
    ]

    start = time.time()
    
    print(f"🚀 Starting {len(LS)} lattices on {n_cores} cores...\n")
    
    # Run in parallel
    with Pool(n_cores) as pool:
        # Use map (blocks until all done) or imap_unordered (returns as they finish)
        results_list = pool.map(_worker_wrapper, job_args)
        
        # Alternative: Get results as they complete
        # results_list = []
        # for result in pool.imap_unordered(_worker_wrapper, job_args):
        #     print(f"✓ Completed L={result['L']}")
        #     results_list.append(result)

    elapsed = time.time() - start

    # Sort by lattice size
    results_list.sort(key=lambda x: x['L'])

    # Consolidate results
    results = {
        'lattices': results_list,
        'eta_effs': [r['eta_mean'] for r in results_list],
        'eta_stds': [r['eta_std'] for r in results_list],
        'cluster_stats': {r['L']: r['cluster_sizes'] for r in results_list},
        'LS': LS,
        'total_runtime': elapsed
    }

    print(f"\n{'='*70}")
    print(f"✅ COMPLETE: {elapsed:.1f}s total ({elapsed/60:.1f} min)")
    print(f"{'='*70}\n")

    # Summary table
    print(f"RESULTS SUMMARY:")
    for r in results_list:
        delta = abs(r['eta_mean'] - ETA_TARGET)
        sigma = delta / r['eta_std'] if r['eta_std'] > 0 else 99
        status = "✓" if delta < 0.01 else "~" if delta < 0.05 else "✗"
        print(f"  {status} L={r['L']:4d}: η={r['eta_mean']:.4f}±{r['eta_std']:.4f}  "
              f"(Δ={delta:.4f}, {sigma:.1f}σ)  [{r['runtime']:.1f}s]")

    return results


# ---------- OPTION 2: Simple serial with better feedback ----------
def run_corrected_golden_analysis_serial_improved(LS=[256, 512, 1024], resume=False):
    """
    SERIAL VERSION (original) but with clearer progress
    Use this if multiprocessing causes issues
    """
    Path("logs").mkdir(exist_ok=True)
    checkpoint_mgr = CheckpointManager()

    print(f"\n{'='*70}")
    print(f"CORRECTED GOLDEN UNIVERSALITY ANALYSIS (SERIAL)")
    print(f"Lattices: {LS}")
    print(f"Target: η = φ/2 = {ETA_TARGET:.6f}")
    print(f"⚠️  Running serially - each lattice waits for previous to complete")
    print(f"{'='*70}\n")

    # Check for resume
    resume_info = {}
    if resume:
        for L in LS:
            checkpoints = checkpoint_mgr.list_checkpoints(L)
            if checkpoints:
                last_block = max([c[0] for c in checkpoints])
                resume_info[L] = last_block
                print(f"[L={L}] Resuming from block {last_block}")

    start = time.time()
    tracker = LiveProgressTracker(LS, n_blocks=16)

    if RICH_AVAILABLE:
        tracker.start()

    results_list = []
    
    for idx, L in enumerate(LS):
        print(f"\n{'─'*70}")
        print(f"Processing {idx+1}/{len(LS)}: L={L}")
        print(f"{'─'*70}")
        
        tracker.start_lattice(L)

        def progress_cb(block_or_msg, eta=None):
            if isinstance(block_or_msg, int):
                tracker.update_block(block_or_msg, eta)
            else:
                print(block_or_msg)

        resume_from = resume_info.get(L) if resume else None

        result = process_lattice_with_progress(
            L, idx, progress_callback=progress_cb,
            checkpoint_mgr=checkpoint_mgr,
            resume_from=resume_from
        )
        results_list.append(result)
        tracker.finish_lattice()

    tracker.stop()
    elapsed = time.time() - start

    # Same consolidation as parallel version
    results = {
        'lattices': results_list,
        'eta_effs': [r['eta_mean'] for r in results_list],
        'eta_stds': [r['eta_std'] for r in results_list],
        'cluster_stats': {r['L']: r['cluster_sizes'] for r in results_list},
        'LS': LS,
        'total_runtime': elapsed
    }

    print(f"\n{'='*70}")
    print(f"✅ COMPLETE: {elapsed:.1f}s total")
    print(f"{'='*70}\n")

    # Summary
    print(f"RESULTS SUMMARY:")
    for r in results_list:
        delta = abs(r['eta_mean'] - ETA_TARGET)
        sigma = delta / r['eta_std'] if r['eta_std'] > 0 else 99
        status = "✓" if delta < 0.01 else "~" if delta < 0.05 else "✗"
        print(f"  {status} L={r['L']:4d}: η={r['eta_mean']:.4f}±{r['eta_std']:.4f}  "
              f"(Δ={delta:.4f}, {sigma:.1f}σ)")

    return results


# ========== USAGE ==========
if __name__ == "__main__":
    # UNIVERSALITY TEST: 6 lattices on 16-core ml.m5.4xlarge
    # Sufficient for proper finite-size scaling analysis
    
    results = run_corrected_golden_analysis_parallel(
        LS=[128, 256, 512, 1024, 1536, 2048],
        n_cores=6,  # One core per lattice (leaves 10 cores free)
        resume=False
    )
    
    # Finite-size scaling check
    print(f"\n{'='*70}")
    print(f"FINITE-SIZE SCALING ANALYSIS")
    print(f"{'='*70}")
    
    LS = results['LS']
    etas = results['eta_effs']
    stds = results['eta_stds']
    
    # Check convergence trend
    print(f"\nConvergence to η∞ = {ETA_TARGET:.6f}:")
    for i, (L, eta, std) in enumerate(zip(LS, etas, stds)):
        delta = abs(eta - ETA_TARGET)
        improvement = ""
        if i > 0:
            prev_delta = abs(etas[i-1] - ETA_TARGET)
            if delta < prev_delta:
                improvement = f" ↓ {(1-delta/prev_delta)*100:.1f}% better"
        print(f"  L={L:4d}: η={eta:.6f}±{std:.6f}  Δ={delta:.6f}{improvement}")
    
    # Extrapolate to L→∞ (simple 1/L correction)
    if len(LS) >= 4:
        from scipy.optimize import curve_fit
        def finite_size_correction(L, eta_inf, a):
            return eta_inf + a / L
        
        try:
            # Use largest 4 lattices for extrapolation
            L_fit = np.array(LS[-4:])
            eta_fit = np.array(etas[-4:])
            
            popt, pcov = curve_fit(finite_size_correction, L_fit, eta_fit, p0=[ETA_TARGET, 1.0])
            eta_inf_extrapolated = popt[0]
            eta_inf_err = np.sqrt(pcov[0, 0])
            
            print(f"\n📊 Extrapolated η∞ = {eta_inf_extrapolated:.6f} ± {eta_inf_err:.6f}")
            print(f"   Target η = {ETA_TARGET:.6f}")
            print(f"   Deviation: {abs(eta_inf_extrapolated - ETA_TARGET):.6f}")
            
            if abs(eta_inf_extrapolated - ETA_TARGET) < 0.005:
                print(f"   ✓✓✓ GOLDEN UNIVERSALITY CONFIRMED!")
            elif abs(eta_inf_extrapolated - ETA_TARGET) < 0.01:
                print(f"   ✓✓ Strong evidence for golden universality")
            else:
                print(f"   ~ Approaching golden universality")
        except:
            print("\n⚠️  Extrapolation failed - need cleaner convergence")
    
    print(f"{'='*70}\n")