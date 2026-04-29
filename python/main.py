#!/usr/bin/env python3
"""Preview a saccade sweep HDF5 file."""

import sys
import h5py
import numpy as np


def preview_h5(path: str) -> None:
    with h5py.File(path, "r") as f:
        event_names = np.array(f["event_names"]).astype(str)
        batch_ids = np.array(f["batch_id"])
        n_events = len(event_names)

        print(f"File: {path}")
        print(f"Events: {n_events}")
        print(f"Batches: {batch_ids.min()} – {batch_ids.max()}")
        print()

        for group_name in f:
            obj = f[group_name]
            if not isinstance(obj, h5py.Group):
                continue

            rates = np.array(obj["rates"])  # shape: (n_events, n_samples)
            n_samples = rates.shape[1]
            print(f"Thread: {group_name}  ({n_events} events × {n_samples} samples)")
            print(f"  {'Event':<40} {'Non-NaN':>8}  {'Mean':>12}  {'Std':>12}")
            print(f"  {'-'*40} {'-'*8}  {'-'*12}  {'-'*12}")

            for i, name in enumerate(event_names):
                row = rates[i]
                valid = row[~np.isnan(row)]
                count = len(valid)
                mean = valid.mean() if count else float("nan")
                std = valid.std() if count else float("nan")
                print(f"  {name:<40} {count:>8}  {mean:>12.4f}  {std:>12.4f}")

            print()


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.h5>", file=sys.stderr)
        sys.exit(1)

    preview_h5(sys.argv[1])


if __name__ == "__main__":
    main()
