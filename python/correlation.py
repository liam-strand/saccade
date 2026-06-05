#!/usr/bin/env python3
"""
Compute pairwise Pearson cross-correlations from saccade sweep HDF5 data.

Cross-batch correlations are valid because:
  1. Each batch's time axis is relative to that batch's first sample, so
     timestep t in any batch represents "t ns into the program's execution"
     (the same program phase regardless of which batch produced the event).
  2. Non-anchor rates are instruction-normalized
     (count/anchor_count * global_ref_rate), making them directly comparable
     across batches.

Outputs correlation.json (event_names, correlation matrix, per-event variance,
co-observation counts, same-batch flag) and a heatmap PNG.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

# Canonical output dir (shared with the analysis scripts), cwd-independent.
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def compute_correlations(
    h5_path: str,
    threshold: float = 0.1,
    min_samples: int = 100,
) -> dict:
    """
    Return a dict suitable for writing to correlation.json.

    Parameters
    ----------
    h5_path:
        Path to sweep_data.h5.
    threshold:
        Pearson |r| values below this are zeroed before PSD projection.
    min_samples:
        Minimum number of co-observed timesteps required to include a pair.
    """
    with h5py.File(h5_path, "r") as f:
        bench_names = list(f.keys())

        # All benchmarks must agree on event set; read from first.
        first = f[bench_names[0]]
        event_names = [
            s.decode() if isinstance(s, bytes) else s
            for s in first["event_names"][:]
        ]
        n_events = len(event_names)

        # Accumulators for pooled Pearson r using the online identity:
        #   r = (n·Σxy − Σx·Σy) / sqrt((n·Σxx − Σx²)(n·Σyy − Σy²))
        # where all sums are over joint-valid timesteps for each pair.
        # Computed via matmul rather than a Python pair loop.
        acc_n   = np.zeros((n_events, n_events), dtype=np.float64)  # co-observation count
        acc_si  = np.zeros((n_events, n_events), dtype=np.float64)  # Σx over joint-valid
        acc_sij = np.zeros((n_events, n_events), dtype=np.float64)  # Σxy
        acc_sii = np.zeros((n_events, n_events), dtype=np.float64)  # Σx² over joint-valid
        is_same_batch = np.zeros((n_events, n_events), dtype=bool)

        # Per-event variance accumulator (Welford)
        ev_count = np.zeros(n_events, dtype=np.int64)
        ev_mean = np.zeros(n_events, dtype=np.float64)
        ev_M2 = np.zeros(n_events, dtype=np.float64)

        for bench_name in bench_names:
            print(f"  {bench_name} ...", file=sys.stderr)
            bg = f[bench_name]
            b_batch_ids = bg["batch_id"][:].tolist()
            b_arr = np.asarray(b_batch_ids, dtype=np.int64)
            same = b_arr[:, None] == b_arr[None, :]
            np.fill_diagonal(same, False)  # diagonal stays False, consistent with current behavior
            is_same_batch |= same
            thread_keys = [k for k in bg.keys() if k.startswith("thread_")]

            for tk in thread_keys:
                rates = bg[tk]["rates"][:]  # (n_events, n_timesteps), float32

                # Per-event variance update (parallel Welford merge — O(n_events) not O(n_samples)).
                for i in range(n_events):
                    valid = rates[i][~np.isnan(rates[i])].astype(np.float64)
                    if len(valid) == 0:
                        continue
                    n_new = len(valid)
                    mean_new = float(valid.mean())
                    M2_new = float(((valid - mean_new) ** 2).sum())
                    n_total = ev_count[i] + n_new
                    delta = mean_new - ev_mean[i]
                    ev_mean[i] = (ev_count[i] * ev_mean[i] + n_new * mean_new) / n_total
                    ev_M2[i] += M2_new + delta * delta * ev_count[i] * n_new / n_total
                    ev_count[i] = n_total

                # Cross-correlation via matmul (all pairs, including cross-batch).
                # NaN → 0 so matmul ignores missing timesteps; mask tracks validity.
                M   = ~np.isnan(rates)
                Mf  = M.astype(np.float64)
                R   = np.where(M, rates, 0.0).astype(np.float64)
                R2  = R * R
                acc_n   += Mf @ Mf.T   # [i,j] = co-observed timestep count
                acc_si  += R  @ Mf.T   # [i,j] = Σ R[i,t] where both valid
                acc_sij += R  @ R.T    # [i,j] = Σ R[i,t]*R[j,t]
                acc_sii += R2 @ Mf.T   # [i,j] = Σ R[i,t]² where both valid

    # Build full symmetric correlation matrix via the online Pearson identity.
    n    = acc_n
    si   = acc_si
    sj   = si.T          # symmetric: Σy over joint-valid = (Σx over joint-valid).T
    sij  = acc_sij
    sii  = acc_sii
    sjj  = sii.T

    numer    = n * sij - si * sj
    denom_i  = n * sii - si  ** 2
    denom_j  = n * sjj - sj  ** 2
    denom_sq = denom_i * denom_j

    valid = (n >= min_samples) & (denom_sq > 0)
    corr  = np.where(
        valid,
        np.clip(numer / np.sqrt(np.where(denom_sq > 0, denom_sq, 1.0)), -1.0, 1.0),
        0.0,
    )
    np.fill_diagonal(corr, 1.0)
    # Threshold weak correlations (off-diagonals only).
    mask = (np.abs(corr) < threshold) & (
        ~np.eye(n_events, dtype=bool)
    )
    corr[mask] = 0.0

    # Project to nearest PSD via eigenvalue clipping.
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize so diagonal stays 1.0.
    d = np.sqrt(np.diag(corr_psd))
    d[d == 0.0] = 1.0
    corr_psd = corr_psd / np.outer(d, d)
    np.fill_diagonal(corr_psd, 1.0)

    # Per-event variance (sample variance from Welford).
    variance = np.where(
        ev_count > 1,
        ev_M2 / (ev_count - 1),
        0.0,
    ).tolist()

    return {
        "event_names": event_names,
        "correlation": corr_psd.tolist(),
        "variance": variance,
        "n_coobserved": acc_n.astype(np.int64).tolist(),
        "is_same_batch": is_same_batch.tolist(),
    }


def plot_heatmap(result: dict, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping heatmap", file=sys.stderr)
        return

    corr = np.array(result["correlation"])

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Event Cross-Correlation (same-batch pairs; PSD-projected)")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Event index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", help="Path to sweep_data.h5")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Zero out |r| < threshold before PSD projection (default: 0.1)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum co-observed timesteps to include a pair (default: 100)",
    )
    parser.add_argument(
        "--out",
        # NOT under results/: this JSON is calibration data consumed via the
        # correlation_path entries in python/config/kf_*.toml, which expect it
        # at the repo-root-relative path python/correlation.json.
        default=str(Path(__file__).resolve().parent / "correlation.json"),
        help="Output JSON path (default: python/correlation.json)",
    )
    parser.add_argument(
        "--heatmap",
        default=str(RESULTS_DIR / "correlation_heatmap.png"),
        help="Output heatmap PNG path (default: results/correlation_heatmap.png)",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.h5} ...")
    result = compute_correlations(args.h5, args.threshold, args.min_samples)

    corr = np.array(result["correlation"])
    n_nonzero_off_diag = int((np.abs(corr) > 1e-9).sum()) - corr.shape[0]
    print(
        f"Computed {corr.shape[0]}x{corr.shape[0]} correlation matrix; "
        f"{n_nonzero_off_diag // 2} non-zero off-diagonal pairs"
    )

    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"Saved {args.out}")

    plot_heatmap(result, args.heatmap)


if __name__ == "__main__":
    main()
