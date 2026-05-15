#!/usr/bin/env python3
"""
Compute pairwise Pearson cross-correlations from saccade sweep HDF5 data.

Events measured in the same sweep batch share a common time axis within each
benchmark run, making their rates directly comparable at each timestep.
Events from different batches were measured in separate runs — their timestep
indices are not co-observed — so only same-batch pairs yield reliable temporal
correlations.

Outputs correlation.json (event_names, correlation matrix, per-event variance,
co-observation counts, same-batch flag) and a heatmap PNG.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


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

        # Accumulators for pooled within-group Pearson r.
        # Subtract per-group means before accumulating so the result reflects
        # within-run temporal correlation, not between-benchmark mean differences.
        # Only upper triangle (i < j); diagonal handled separately.
        sum_xy = np.zeros((n_events, n_events), dtype=np.float64)
        sum_x2 = np.zeros((n_events, n_events), dtype=np.float64)
        sum_y2 = np.zeros((n_events, n_events), dtype=np.float64)
        pair_count = np.zeros((n_events, n_events), dtype=np.int64)
        is_same_batch = np.zeros((n_events, n_events), dtype=bool)

        # Per-event variance accumulator (Welford)
        ev_count = np.zeros(n_events, dtype=np.int64)
        ev_mean = np.zeros(n_events, dtype=np.float64)
        ev_M2 = np.zeros(n_events, dtype=np.float64)

        for bench_name in bench_names:
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

                # Cross-correlation: same-batch pairs only.
                # Group event indices by batch so we only iterate within-batch.
                by_batch: dict[int, list[int]] = defaultdict(list)
                for idx, bid in enumerate(b_batch_ids):
                    by_batch[bid].append(idx)

                for bid, idxs in by_batch.items():
                    if len(idxs) < 2:
                        continue
                    # For each pair within this batch:
                    for ii, i in enumerate(idxs):
                        valid_i = ~np.isnan(rates[i])
                        if not valid_i.any():
                            continue
                        for j in idxs[ii + 1 :]:
                            valid_j = ~np.isnan(rates[j])
                            both = valid_i & valid_j
                            n = int(both.sum())
                            if n < min_samples:
                                continue
                            xi = rates[i, both].astype(np.float64)
                            xj = rates[j, both].astype(np.float64)
                            # Subtract per-group means so accumulated sums reflect within-group
                            # (temporal) correlation only.
                            xi -= xi.mean()
                            xj -= xj.mean()
                            lo, hi = min(i, j), max(i, j)
                            sum_xy[lo, hi] += np.dot(xi, xj)
                            sum_x2[lo, hi] += np.dot(xi, xi)
                            sum_y2[lo, hi] += np.dot(xj, xj)
                            pair_count[lo, hi] += n

    # Build full symmetric correlation matrix.
    corr = np.eye(n_events, dtype=np.float64)
    for i in range(n_events):
        for j in range(i + 1, n_events):
            n = pair_count[i, j]
            if n < min_samples:
                continue
            denom_sq = sum_x2[i, j] * sum_y2[i, j]
            if denom_sq <= 0.0:
                continue
            r = float(np.clip(sum_xy[i, j] / np.sqrt(denom_sq), -1.0, 1.0))
            corr[i, j] = r
            corr[j, i] = r

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
        "n_coobserved": pair_count.tolist(),
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
        default="python/correlation.json",
        help="Output JSON path (default: python/correlation.json)",
    )
    parser.add_argument(
        "--heatmap",
        default="python/correlation_heatmap.png",
        help="Output heatmap PNG path",
    )
    args = parser.parse_args()

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
