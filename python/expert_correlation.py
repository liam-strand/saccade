#!/usr/bin/env python3
"""
expert_correlation.py — generate an expert-system correlation matrix for saccade's
Kalman filter estimator.

python/correlation.py produces within-batch Pearson correlations from sweep_data.h5.
Because the sweep rotates ~4 events per batch, many important relationships between
events in different batches are invisible to the data-driven approach.  Several of the
most important execution counters (ex_ret_instr, ls_not_halted_cyc, ls_dc_accesses,
op_cache_hit_miss.all_op_cache_accesses, ic_fw32, de_dis_cops_from_decoder.*) are in
singleton-ish batches with no meaningful same-batch partners, so the Kalman filter can
never propagate information across them without expert seeding.

This script merges the data-driven correlations with expert-defined cross-batch
relationships grounded in AMD Zen micro-architecture semantics, then iteratively
projects to PSD and re-applies the |r|≥0.1 threshold until stable.

Output: python/correlation_expert.json (same schema as correlation.json).
"""

import json
import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Load data-driven base
# ---------------------------------------------------------------------------
base_path = Path(__file__).parent / "correlation.json"
with open(base_path) as f:
    base = json.load(f)

names = base["event_names"]
n = len(names)
name_to_idx = {name: i for i, name in enumerate(names)}

corr_data = np.array(base["correlation"], dtype=float)
var        = np.array(base["variance"], dtype=float)
n_co       = np.array(base["n_coobserved"], dtype=float)
is_sb      = (n_co > 0) | (n_co.T > 0)                  # same-batch proxy (n_co is asymmetric)

# ---------------------------------------------------------------------------
# Expert cross-batch correlations (AMD Zen semantics)
# ---------------------------------------------------------------------------
# All entries here are for pairs where is_sb[i,j] = False, i.e. events that
# were never simultaneously sampled and therefore have r=0 in correlation.json.
#
# r values are conservative: we prefer missing a real correlation over injecting
# a spurious one.  The PSD projection will slightly reduce these; diagnostic
# output at the end shows the final values.
#
# Organised by cluster so the reasoning is traceable.

EXPERT = [
    # =========================================================
    # Strategy: only add cross-batch connections for events that have no
    # significant same-batch partners (row_sum < 0.5 in the symmetric
    # is_sb matrix).  Adding cross-batch connections to events that are
    # already in tight same-batch clusters (ex_ret_instr/ops/brn/cond,
    # ic_tag_hit_miss.all, ...) causes PSD projection to erode those
    # same-batch pairs, which is worse than having no cross-batch info.
    #
    # Safe singleton-batch events used here:
    #   macro_ops_retired, ls_not_halted_cyc, ls_dc_accesses,
    #   ls_alloc_mab_count, op_cache_hit_miss.all_op_cache_accesses,
    #   ic_fw32, de_dis_cops_from_decoder.disp_op_type.any_{integer,fp}_dispatch,
    #   bp_l2_btb_correct, bp_l1_btb_correct
    # =========================================================

    # =========================================================
    # Cluster A — Instruction execution throughput
    # macro_ops_retired and ls_not_halted_cyc serve as proxies for the
    # un-connectable ex_ret_instr/ex_ret_ops (those have tight same-batch
    # partners at r≈0.93–0.98 that would be eroded).
    # =========================================================

    # Cycles active ↔ macro-ops retired: both scale with "work done per quantum".
    ("macro_ops_retired", "ls_not_halted_cyc",   0.65),

    # Dispatch events tightly coupled to retire (one pipeline stage earlier).
    ("macro_ops_retired", "de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch", 0.80),
    ("ls_not_halted_cyc", "de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch", 0.72),

    # Op cache provides most fetch bandwidth; accesses scale with throughput.
    ("macro_ops_retired", "op_cache_hit_miss.all_op_cache_accesses",  0.75),
    ("ls_not_halted_cyc", "op_cache_hit_miss.all_op_cache_accesses",  0.72),

    # FP dispatch: workload-type dependent, weaker coupling.
    ("macro_ops_retired", "de_dis_cops_from_decoder.disp_op_type.any_fp_dispatch",      0.45),
    ("de_dis_cops_from_decoder.disp_op_type.any_integer_dispatch",
     "de_dis_cops_from_decoder.disp_op_type.any_fp_dispatch",                           0.40),

    # =========================================================
    # Cluster B — Memory access
    # DC accesses and MAB allocation are tightly coupled; both also
    # correlate with overall execution activity.
    # =========================================================
    ("ls_dc_accesses",    "ls_alloc_mab_count",   0.80),
    ("ls_dc_accesses",    "macro_ops_retired",     0.65),
    ("ls_dc_accesses",    "ls_not_halted_cyc",    0.60),
    ("ls_alloc_mab_count","macro_ops_retired",     0.60),
    ("ls_alloc_mab_count","ls_not_halted_cyc",    0.60),

    # =========================================================
    # Cluster D — Instruction cache fetch
    # ic_fw32 (32-byte fetch windows) is a singleton batch; it scales
    # with instruction throughput and op-cache bandwidth.
    # =========================================================
    ("ic_fw32",           "macro_ops_retired",    0.75),
    ("ic_fw32",           "ls_not_halted_cyc",   0.70),
    ("ic_fw32",           "op_cache_hit_miss.all_op_cache_accesses", 0.72),

    # =========================================================
    # Cluster E — L2 BTB corrections
    # bp_l2_btb_correct is in a singleton batch; correlates with other
    # branch-prediction counters.
    # =========================================================
    ("bp_l2_btb_correct", "bp_l1_btb_correct",   0.50),
    ("bp_l2_btb_correct", "bp_dyn_ind_pred",      0.35),
]

# ---------------------------------------------------------------------------
# Build expert matrix
# ---------------------------------------------------------------------------
expert_r = np.zeros((n, n))
bad = False
for name_i, name_j, r in EXPERT:
    i = name_to_idx.get(name_i)
    j = name_to_idx.get(name_j)
    if i is None:
        print(f"WARNING: unknown event '{name_i}'", file=sys.stderr)
        bad = True
        continue
    if j is None:
        print(f"WARNING: unknown event '{name_j}'", file=sys.stderr)
        bad = True
        continue
    if is_sb[i, j]:
        # Same batch: data already has this pair; expert should not override.
        print(f"NOTE: {name_i} <-> {name_j} is same-batch; skipping expert override",
              file=sys.stderr)
    expert_r[i, j] = r
    expert_r[j, i] = r

if bad:
    sys.exit(1)

# Merge: same-batch pairs come from data; cross-batch pairs come from expert.
# Entries where neither data nor expert have information stay at 0.
merged = corr_data.copy()
cross = ~is_sb
np.fill_diagonal(cross, False)
merged[cross] = expert_r[cross]
np.fill_diagonal(merged, 1.0)

# Symmetrise (should already be symmetric, but be safe).
merged = 0.5 * (merged + merged.T)
np.fill_diagonal(merged, 1.0)

# ---------------------------------------------------------------------------
# Iterative PSD projection + threshold
#
# One round of PSD-clip then hard-threshold can destroy PSD.  We iterate:
#   1. Clip negative eigenvalues → PSD
#   2. Renormalise diagonal to 1 (→ correlation matrix form)
#   3. Zero entries below |r| < THRESHOLD that belong to neither data nor expert
#   4. Repeat until stable (usually 2–3 iterations)
# ---------------------------------------------------------------------------
THRESHOLD = 0.1
ITERS = 8

# Mask of entries that should be kept (data or expert specified them).
specified = (np.abs(corr_data) >= THRESHOLD) | (np.abs(expert_r) >= THRESHOLD)
np.fill_diagonal(specified, True)

def psd_project(C):
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 0.0)
    P = eigvecs @ np.diag(eigvals) @ eigvecs.T
    diag = np.sqrt(np.diag(P))
    diag[diag == 0] = 1.0
    P /= np.outer(diag, diag)
    np.fill_diagonal(P, 1.0)
    return P

C = merged.copy()
for it in range(ITERS):
    C_prev = C.copy()
    C = psd_project(C)
    # Zero out entries that were never specified, if they dropped below threshold.
    unspecified_small = (~specified) & (np.abs(C) < THRESHOLD)
    C[unspecified_small] = 0.0
    # Symmetrise after zeroing.
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    delta = np.max(np.abs(C - C_prev))
    if delta < 5e-6:
        print(f"Converged after {it + 1} iteration(s) (max change {delta:.2e})")
        break
else:
    print(f"Did not converge after {ITERS} iterations (last max change {delta:.2e})")

# Final PSD projection (no threshold after this).
final = psd_project(C)
np.fill_diagonal(final, 1.0)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
min_eig = np.min(np.linalg.eigvalsh(final))
print(f"\nMinimum eigenvalue of final matrix: {min_eig:.3e}  (should be >= -1e-10)")
if min_eig < -1e-8:
    print("WARNING: matrix is not numerically PSD!", file=sys.stderr)

nz_data   = (np.abs(corr_data) >= THRESHOLD).sum() - n
nz_expert = (np.abs(final) >= THRESHOLD).sum() - n
print(f"Non-zero off-diagonal pairs: {nz_data // 2} (data-only) -> {nz_expert // 2} (expert merge)")

print("\nExpert cross-batch pairs — target vs final r:")
print(f"  {'Event A':<55} {'Event B':<55} {'target':>7} {'final':>7} {'delta':>7}")
print(f"  {'-'*55} {'-'*55} {'-'*7} {'-'*7} {'-'*7}")
n_degraded = 0
for name_i, name_j, r_target in sorted(EXPERT, key=lambda x: -abs(x[2])):
    i = name_to_idx.get(name_i)
    j = name_to_idx.get(name_j)
    if i is None or j is None:
        continue
    if is_sb[i, j]:
        continue
    r_final = final[i, j]
    delta = r_final - r_target
    flag = " <<" if abs(delta) > 0.10 else ""
    print(f"  {name_i:<55} {name_j:<55} {r_target:+.3f}  {r_final:+.3f}  {delta:+.3f}{flag}")
    if abs(delta) > 0.10:
        n_degraded += 1

if n_degraded:
    print(f"\nWARNING: {n_degraded} expert pair(s) degraded by >0.10 after PSD projection.",
          file=sys.stderr)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = {
    "event_names":  names,
    "correlation":  final.tolist(),
    "variance":     var.tolist(),
    "n_coobserved": n_co.tolist(),
    "is_same_batch": is_sb.tolist(),
}

out_path = Path(__file__).parent / "correlation_expert.json"
with open(out_path, "w") as f:
    json.dump(out, f)

print(f"\nSaved {out_path}")
