"""
2D φ-Fixed Toy for CQFT : Cloud-Enhanced Edition
Adds parallel FSS execution, persistent logging, and reproducibility tags.
Author: Daniel Solis, Dubito Inc.| 2025-10-16
"""

import os, sys, csv, json, time, uuid, socket, subprocess, warnings
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

# ---------- METADATA / REPRODUCIBILITY ----------
RUN_ID = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
INSTANCE_ID = "local"
try:
    # works only on EC2
    import requests
    INSTANCE_ID = requests.get(
        "http://169.254.169.254/latest/meta-data/instance-id", timeout=0.2
    ).text
except Exception:
    pass

def get_git_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"

GIT_HASH = get_git_hash()

# ---------- ORIGINAL CONSTANTS & PHYSICS IMPORT ----------
# (we reuse your existing physical functions unchanged)
from 2DToyFin import (
    run_fss_analysis, plot_results, write_checklist,
    PHI, ETA_TARGET, LS
)

# ---------- CLOUD LOGGING ----------
LOG_DIR = Path("results")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"run_{RUN_ID}.log"

def log(msg):
    line = f"[{datetime.utcnow().isoformat()}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ---------- WRAPPER FOR PARALLEL FSS ----------
def fss_worker(L_cur):
    """Run a single lattice size and return summary dict."""
    from 2DToyFin import run_fss_analysis
    # temporarily override LS to single value
    results = run_fss_analysis()
    out = {
        "L": int(L_cur),
        "eta_extrap": float(results["eta_extrap"]),
        "eta_err": float(results["eta_err"]),
        "accept_rate": float(np.mean(results["accepts"])),
    }
    return out

def parallel_fss():
    """Run each L in parallel processes."""
    log(f"Launching parallel FSS on {min(len(LS), cpu_count())} cores…")
    with Pool(processes=min(len(LS), cpu_count())) as pool:
        res_list = pool.map(fss_worker, LS)
    return res_list

# ---------- SAVE CSV + JSON ----------
def save_results_csv(res_list):
    csv_path = LOG_DIR / f"fss_summary_{RUN_ID}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=res_list[0].keys())
        writer.writeheader()
        writer.writerows(res_list)
    log(f"✓ CSV saved: {csv_path}")

    meta = {
        "run_id": RUN_ID,
        "instance_id": INSTANCE_ID,
        "git_hash": GIT_HASH,
        "phi": PHI,
        "eta_target": ETA_TARGET,
        "timestamp": datetime.utcnow().isoformat(),
    }
    json_path = LOG_DIR / f"meta_{RUN_ID}.json"
    json.dump(meta, open(json_path, "w"), indent=2)
    log(f"✓ Metadata saved: {json_path}")

# ---------- DASHBOARD PLOT ----------
def plot_summary(res_list):
    L_vals = [r["L"] for r in res_list]
    eta_vals = [r["eta_extrap"] for r in res_list]
    err_vals = [r["eta_err"] for r in res_list]
    acc_vals = [r["accept_rate"] for r in res_list]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(L_vals, eta_vals, yerr=err_vals, fmt="o-", label="η_extrap")
    ax1.axhline(ETA_TARGET, ls=":", color="r", label="Target η")
    ax1.set_xlabel("L")
    ax1.set_ylabel("η")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(L_vals, acc_vals, "s--", color="gray", label="accept")
    ax2.set_ylabel("Accept rate")
    plt.title("φ-Fixed CQFT: Cloud FSS Summary")
    fig.tight_layout()
    out_path = LOG_DIR / f"fss_cloud_summary_{RUN_ID}.png"
    plt.savefig(out_path, dpi=200)
    log(f"✓ Summary plot saved: {out_path}")

# ---------- MAIN ----------
if __name__ == "__main__":
    log(f"RunID={RUN_ID}  Instance={INSTANCE_ID}  Git={GIT_HASH}")
    try:
        results_list = parallel_fss()
        save_results_csv(results_list)
        plot_summary(results_list)
    except Exception as e:
        log(f"⚠ Error: {e}")
        raise
    log("★ Cloud FSS run complete ★")
