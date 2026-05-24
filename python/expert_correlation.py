#!/usr/bin/env python3
"""
expert_correlation.py — expert-augmented correlation matrix for saccade's Kalman estimator.

Builds on python/correlation.py's analytical output with two additional signals:

1. Benchmark-fingerprint correlations (from sweep_data.h5):
   For each event, compute its mean rate per benchmark → 8-element vector.
   Pairs with highly similar benchmark fingerprints are architecturally related
   even if their within-run temporal correlation is weak (due to alignment jitter
   across sweep runs, sparse event counts, or low temporal variance).
   Applied only to cross-batch pairs still below threshold.

2. Expanded AMD Zen expert priors:
   Manually specified r values for pairs that are architecturally tightly coupled
   but fall below the analytical threshold for structural reasons (sparsity, alias
   event families, etc.).

Merge order (highest to lowest trust):
  same-batch analytical r (never overridden)
  > cross-batch analytical r (|r| >= threshold)
  > fingerprint r (|r| >= fingerprint_threshold, n=8 benchmarks)
  > expert prior

Output: python/expert2_correlation.json (same schema as correlation.json).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Expert priors (AMD Zen micro-architecture semantics)
#
# Entries are (event_a, event_b, r_prior).  Only applied to cross-batch pairs
# where both the analytical and fingerprint layers are still below threshold.
# r values are conservative — the PSD projection will reduce them further.
# ---------------------------------------------------------------------------

EXPERT = [
    # =========================================================
    # Cluster A — Instruction fetch / op-cache front-end
    # Events in different batches that measure the same fetch bandwidth.
    # =========================================================
    ("ic_fw32", "op_cache_hit_miss.all_op_cache_accesses",          0.44),
    ("ic_fw32", "ic_tag_hit_miss.all_instruction_cache_accesses",   0.60),
    ("op_cache_hit_miss.op_cache_miss",
     "ic_oc_mode_switch.ic_oc_mode_switch",                         0.65),
    ("op_cache_hit_miss.op_cache_hit",
     "ic_tag_hit_miss.instruction_cache_hit",                       0.55),
    ("ic_cache_fill_l2", "l2_cache_req_stat.ic_access_in_l2",       0.70),
    ("ic_cache_fill_sys", "l2_cache_req_stat.ic_fill_miss",         0.72),

    # =========================================================
    # Cluster B — L1/L2 data cache and miss-address buffer
    # Different perf event families that measure the same cache-pressure pathway.
    # =========================================================
    ("ls_alloc_mab_count",              "ls_dc_accesses",           0.50),
    ("ls_alloc_mab_count",              "macro_ops_retired",        0.34),
    ("ls_mab_alloc.all_allocations",    "ls_dc_accesses",           0.45),
    ("ls_mab_alloc.load_store_allocations", "ls_dc_accesses",       0.45),
    ("ls_dispatch.ld_dispatch",         "ls_dc_accesses",           0.45),
    ("l2_fill_pending.l2_fill_busy",
     "l2_latency.l2_cycles_waiting_on_fills",                       0.78),
    ("l2_cache_accesses_from_dc_misses", "ls_dc_accesses",          0.40),

    # =========================================================
    # Cluster C — L1 data-cache fill source aliases
    # Newer perf names (l1_data_cache_fills_*) vs older ls_any_fills_from_sys.*
    # and ls_dmnd_fills_from_sys.* measure the same physical fills.
    # =========================================================
    ("l1_data_cache_fills_from_within_same_ccx",
     "ls_any_fills_from_sys.int_cache",                             0.88),
    ("l1_data_cache_fills_from_external_ccx_cache",
     "ls_any_fills_from_sys.ext_cache_local",                       0.88),
    ("l1_data_cache_fills_from_memory",
     "ls_any_fills_from_sys.mem_io_local",                          0.88),
    ("l1_data_cache_fills_from_remote_node",
     "ls_any_fills_from_sys.ext_cache_remote",                      0.85),
    ("l1_data_cache_fills_all",         "ls_dc_accesses",           0.45),
    ("all_data_cache_accesses",         "ls_dc_accesses",           0.50),

    # =========================================================
    # Cluster D — Branch prediction (sparse/rare events)
    # Rare-event counters where alignment jitter and sparsity hurt Pearson r.
    # =========================================================
    ("bp_dyn_ind_pred",         "bp_l2_btb_correct",                0.35),
    ("ex_ret_brn_ind_misp",     "bp_dyn_ind_pred",                  0.60),
    ("ex_ret_brn_misp",         "bp_l2_btb_correct",                0.50),
    ("bp_snp_re_sync",          "bp_de_redirect",                   0.45),

    # =========================================================
    # Cluster E — FP/SIMD pipeline (workload-type dependent)
    # FP events appear only in FP-heavy workloads; within-run temporal signal
    # is sparse, but across workloads the pattern is consistent.
    # =========================================================
    ("fp_ret_sse_avx_ops.all",  "fpu_pipe_assignment.total",        0.82),
    ("fp_ret_sse_avx_ops.mult_flops",   "fpu_pipe_assignment.total", 0.68),
    ("fp_ret_sse_avx_ops.add_sub_flops","fpu_pipe_assignment.total", 0.65),
    ("sse_avx_stalls",          "fp_ret_sse_avx_ops.all",           0.62),
    ("sse_avx_stalls",          "fpu_pipe_assignment.total",        0.58),
    ("ex_div_busy",             "ex_div_count",                     0.90),

    # =========================================================
    # Cluster F — TLB hierarchy (L1 → L2 → tablewalker causal chain)
    # =========================================================
    ("l1_dtlb_misses",          "ls_l1_d_tlb_miss.all",             0.88),
    ("l2_dtlb_misses",          "ls_tablewalker.dside",             0.78),
    ("l2_itlb_misses",          "ls_tablewalker.iside",             0.75),
    ("ls_tablewalker.dside",    "ls_tablewalker.dc_type0",          0.72),
    ("ls_tablewalker.iside",    "ls_tablewalker.ic_type0",          0.72),
    ("l2_itlb_misses",          "bp_l1_tlb_miss_l2_tlb_miss.if4k", 0.65),
    ("l2_dtlb_misses",          "ls_l1_d_tlb_miss.all",            0.60),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def psd_project(C: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 0.0)
    P = eigvecs @ np.diag(eigvals) @ eigvecs.T
    diag = np.sqrt(np.diag(P))
    diag[diag == 0] = 1.0
    P /= np.outer(diag, diag)
    np.fill_diagonal(P, 1.0)
    return P


def compute_fingerprints(h5_path: str, event_names: list[str]) -> np.ndarray:
    """
    Return (n_events, n_benchmarks) matrix of per-benchmark mean rates.
    Rows correspond to event_names order.  NaN where an event was never
    observed in a benchmark.
    """
    with h5py.File(h5_path, "r") as f:
        bench_names = list(f.keys())
        n_events = len(event_names)
        n_bench = len(bench_names)
        mean_b = np.full((n_events, n_bench), np.nan)

        for b_idx, bench in enumerate(bench_names):
            bg = f[bench]
            h5_names = [
                s.decode() if isinstance(s, bytes) else s
                for s in bg["event_names"][:]
            ]
            name_to_row = {name: i for i, name in enumerate(h5_names)}

            thread_keys = [k for k in bg.keys() if k.startswith("thread_")]
            # Accumulate per-event sum and count across all threads.
            total_sum = np.zeros(n_events)
            total_cnt = np.zeros(n_events, dtype=np.int64)

            for tk in thread_keys:
                rates = bg[tk]["rates"][:]  # (n_h5_events, n_timesteps)
                for ev_idx, ev_name in enumerate(event_names):
                    row_h5 = name_to_row.get(ev_name)
                    if row_h5 is None:
                        continue
                    r = rates[row_h5]
                    valid = r[~np.isnan(r)]
                    total_sum[ev_idx] += float(valid.sum())
                    total_cnt[ev_idx] += len(valid)

            with np.errstate(invalid="ignore"):
                mean_b[:, b_idx] = np.where(
                    total_cnt > 0, total_sum / total_cnt, np.nan
                )

    return mean_b


def fingerprint_correlations(mean_b: np.ndarray) -> np.ndarray:
    """
    Return (n_events, n_events) Pearson r matrix computed from benchmark
    fingerprints (the rows of mean_b).  Rows with fewer than 3 valid
    benchmarks get r=0 with all other events.
    """
    n_events, n_bench = mean_b.shape
    # Z-score each event's benchmark vector (across benchmarks).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(mean_b, axis=1, ddof=1)
        mean = np.nanmean(mean_b, axis=1)
    valid_count = (~np.isnan(mean_b)).sum(axis=1)

    z = np.full_like(mean_b, np.nan)
    for i in range(n_events):
        if valid_count[i] >= 3 and std[i] > 1e-30:
            z[i] = (mean_b[i] - mean[i]) / std[i]

    # Pairwise Pearson r via dot products on z-scored rows (ignoring NaN).
    fp_r = np.zeros((n_events, n_events))
    for i in range(n_events):
        if valid_count[i] < 3:
            continue
        for j in range(i + 1, n_events):
            if valid_count[j] < 3:
                continue
            # Use only benchmarks where both are valid.
            mask = ~np.isnan(z[i]) & ~np.isnan(z[j])
            n_valid = mask.sum()
            if n_valid < 3:
                continue
            zi = z[i][mask]
            zj = z[j][mask]
            # Re-standardise on the joint-valid subset.
            si = zi.std(ddof=1)
            sj = zj.std(ddof=1)
            if si < 1e-30 or sj < 1e-30:
                continue
            r = float(np.dot((zi - zi.mean()) / si, (zj - zj.mean()) / sj) / (n_valid - 1))
            r = float(np.clip(r, -1.0, 1.0))
            fp_r[i, j] = r
            fp_r[j, i] = r

    np.fill_diagonal(fp_r, 1.0)
    return fp_r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "h5",
        nargs="?",
        default="sweep_data.h5",
        help="Path to sweep_data.h5 (default: python/sweep_data.h5)",
    )
    parser.add_argument(
        "--base",
        default="correlation.json",
        help="Base analytical correlation JSON (default: python/correlation.json)",
    )
    parser.add_argument(
        "--out",
        default="expert2_correlation.json",
        help="Output JSON path (default: python/expert2_correlation.json)",
    )
    parser.add_argument(
        "--fingerprint-threshold",
        type=float,
        default=0.90,
        help="Min |fingerprint r| to apply as prior (default: 0.90); "
             "α=0.05 critical value for n=8 is ≈0.707",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Weak-correlation threshold, consistent with correlation.py (default: 0.1)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Max PSD projection iterations (default: 20)",
    )
    args = parser.parse_args()

    THRESHOLD = args.threshold
    FP_THRESHOLD = args.fingerprint_threshold
    ITERS = args.iters

    # ------------------------------------------------------------------
    # Load base analytical correlations
    # ------------------------------------------------------------------
    print(f"Loading {args.base} ...", file=sys.stderr)
    with open(args.base) as f:
        base = json.load(f)

    names = base["event_names"]
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    corr_data = np.array(base["correlation"], dtype=float)
    var        = np.array(base["variance"], dtype=float)
    n_co       = np.array(base["n_coobserved"], dtype=float)
    is_sb      = np.array(base["is_same_batch"], dtype=bool)

    # ------------------------------------------------------------------
    # Compute benchmark fingerprint correlations from h5
    # ------------------------------------------------------------------
    print(f"Loading {args.h5} for benchmark fingerprints ...", file=sys.stderr)
    mean_b = compute_fingerprints(args.h5, names)
    fp_r = fingerprint_correlations(mean_b)

    # ------------------------------------------------------------------
    # Build expert prior matrix
    # ------------------------------------------------------------------
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
            print(
                f"NOTE: {name_i} <-> {name_j} is same-batch; skipping expert override",
                file=sys.stderr,
            )
            continue
        expert_r[i, j] = r
        expert_r[j, i] = r

    if bad:
        sys.exit(1)

    # ------------------------------------------------------------------
    # Merge: analytical → fingerprint → expert
    # ------------------------------------------------------------------
    merged = corr_data.copy()
    cross = ~is_sb
    np.fill_diagonal(cross, False)

    # Layer 1: fingerprint fills cross-batch weak pairs.
    cross_weak = cross & (np.abs(corr_data) < THRESHOLD)
    fp_strong  = np.abs(fp_r) >= FP_THRESHOLD
    fill_fp    = cross_weak & fp_strong
    merged[fill_fp] = fp_r[fill_fp]

    n_fp_filled = int(fill_fp.sum()) // 2
    print(f"Fingerprint layer filled {n_fp_filled} cross-batch pairs (|fp_r| ≥ {FP_THRESHOLD})")

    # Layer 2: expert fills remaining weak cross-batch pairs.
    still_weak = cross & (np.abs(merged) < THRESHOLD)
    fill_exp   = still_weak & (np.abs(expert_r) >= THRESHOLD)
    merged[fill_exp] = expert_r[fill_exp]

    n_exp_filled = int(fill_exp.sum()) // 2
    print(f"Expert layer filled {n_exp_filled} additional cross-batch pairs")

    np.fill_diagonal(merged, 1.0)
    merged = 0.5 * (merged + merged.T)
    np.fill_diagonal(merged, 1.0)

    # ------------------------------------------------------------------
    # Iterative PSD projection + threshold
    # ------------------------------------------------------------------
    # Mask of entries that should be preserved (data, fingerprint, or expert).
    specified = (
        (np.abs(corr_data) >= THRESHOLD)
        | fp_strong
        | (np.abs(expert_r) >= THRESHOLD)
    )
    np.fill_diagonal(specified, True)

    C = merged.copy()
    for it in range(ITERS):
        C_prev = C.copy()
        C = psd_project(C)
        unspecified_small = (~specified) & (np.abs(C) < THRESHOLD)
        C[unspecified_small] = 0.0
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 1.0)
        delta = float(np.max(np.abs(C - C_prev)))
        if delta < 5e-6:
            print(f"Converged after {it + 1} iteration(s) (max change {delta:.2e})")
            break
    else:
        print(f"Did not converge after {ITERS} iterations (last max change {delta:.2e})")

    final = psd_project(C)
    np.fill_diagonal(final, 1.0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    min_eig = float(np.min(np.linalg.eigvalsh(final)))
    print(f"\nMinimum eigenvalue: {min_eig:.3e}  (should be ≥ −1e-10)")
    if min_eig < -1e-8:
        print("WARNING: matrix is not numerically PSD!", file=sys.stderr)

    nz_base  = (np.abs(corr_data) >= THRESHOLD).sum() - n
    nz_final = (np.abs(final)     >= THRESHOLD).sum() - n
    print(
        f"Non-zero off-diagonal pairs: {nz_base // 2} (base) "
        f"→ {nz_final // 2} (expert2)"
    )

    # Fingerprint pairs that actually changed the matrix.
    fp_pairs = list(zip(*np.where(fill_fp & (final != corr_data))))
    fp_pairs = [(i, j) for i, j in fp_pairs if i < j]
    fp_pairs.sort(key=lambda ij: -abs(fp_r[ij]))
    if fp_pairs:
        print(f"\nFingerprint-derived cross-batch pairs (top {min(20, len(fp_pairs))}):")
        print(f"  {'Event A':<55} {'Event B':<55} {'fp_r':>6} {'base_r':>7} {'final':>7}")
        print(f"  {'-'*55} {'-'*55} {'-'*6} {'-'*7} {'-'*7}")
        for i, j in fp_pairs[:20]:
            print(
                f"  {names[i]:<55} {names[j]:<55} "
                f"{fp_r[i,j]:+.3f}  {corr_data[i,j]:+.3f}  {final[i,j]:+.3f}"
            )
    else:
        print("\nNo fingerprint-derived pairs filled.")

    # Expert pairs that filled a gap.
    exp_filled_pairs = [(i, j) for i, j in zip(*np.where(fill_exp)) if i < j]
    if exp_filled_pairs:
        print(f"\nExpert-derived cross-batch pairs — target vs final r:")
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
            if not fill_exp[i, j]:
                continue
            r_final = final[i, j]
            delta = r_final - r_target
            flag = " <<" if abs(delta) > 0.10 else ""
            print(
                f"  {name_i:<55} {name_j:<55} "
                f"{r_target:+.3f}  {r_final:+.3f}  {delta:+.3f}{flag}"
            )
            if abs(delta) > 0.10:
                n_degraded += 1
        if n_degraded:
            print(
                f"\nWARNING: {n_degraded} expert pair(s) degraded by >0.10 after PSD projection.",
                file=sys.stderr,
            )
    else:
        print("\nNo expert pairs filled any gaps (all already covered by analytical/fingerprint).")

    # All new pairs: above threshold in final but not in base.
    new_mask = (np.abs(final) >= THRESHOLD) & (np.abs(corr_data) < THRESHOLD)
    new_pairs = [(i, j) for i, j in zip(*np.where(new_mask)) if i < j]
    new_pairs.sort(key=lambda ij: -abs(final[ij]))
    n_new = len(new_pairs)
    print(f"\nAll new pairs in expert2 vs base ({n_new} total), sorted by |final r|:")
    print(f"  {'Event A':<55} {'Event B':<55} {'final':>7}  source")
    print(f"  {'-'*55} {'-'*55} {'-'*7}  ------")
    for i, j in new_pairs:
        if fill_fp[i, j]:
            src = "fingerprint"
        elif fill_exp[i, j]:
            src = "expert"
        else:
            src = "psd-propagated"
        print(
            f"  {names[i]:<55} {names[j]:<55} {final[i,j]:+.3f}  {src}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = {
        "event_names":   names,
        "correlation":   final.tolist(),
        "variance":      var.tolist(),
        "n_coobserved":  n_co.tolist(),
        "is_same_batch": is_sb.tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
