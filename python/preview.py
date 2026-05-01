#!/usr/bin/env python3
"""Preview a saccade aggregated sweep HDF5 file (collect.py output)."""

import sys
import h5py
import numpy as np


def preview_benchmark(group: h5py.Group) -> None:
    event_names = np.array(group["event_names"]).astype(str)
    batch_ids = np.array(group["batch_id"])
    n_events = len(event_names)

    print(f"=== {group.name.lstrip('/')} ===")
    print(f"Events: {n_events}")
    print(f"Batches: {batch_ids.min()} – {batch_ids.max()}")
    print()

    for thread_name in group:
        obj = group[thread_name]
        if not isinstance(obj, h5py.Group):
            continue

        rates = np.array(obj["rates"])
        n_samples = rates.shape[1]
        print(f"Thread: {thread_name}  ({n_events} events × {n_samples} samples)")
        print(f"  {'Event':<40} {'Non-NaN':>8}  {'Mean':>12}  {'Std':>12}")
        print(f"  {'-' * 40} {'-' * 8}  {'-' * 12}  {'-' * 12}")

        for i, name in enumerate(event_names):
            row = rates[i]
            valid = row[~np.isnan(row)]
            count = len(valid)
            mean = valid.mean() if count else float("nan")
            std = valid.std() if count else float("nan")
            print(f"  {name:<40} {count:>8}  {mean:>12.4f}  {std:>12.4f}")

        print()


def preview_h5(path: str) -> None:
    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        bench_names = [k for k in f if isinstance(f[k], h5py.Group) and "event_names" in f[k]]
        print(f"Benchmarks: {len(bench_names)}")
        print()

        for name in bench_names:
            preview_benchmark(f[name])


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.h5>", file=sys.stderr)
        sys.exit(1)

    preview_h5(sys.argv[1])


if __name__ == "__main__":
    main()
