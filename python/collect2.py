#!/usr/bin/env python3
"""Collect ground truth evaluation sweep data across NPB and SPEC CPU2017 benchmarks into a single HDF5 file."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
from tqdm import tqdm

NPB_DIR = Path("/tank/yhe7443/benchmarks/NPB3.3.1/NPB3.3-SER/bin")
SPEC_ROOT = Path("/tank/yhe7443/benchmarks/SPEC2017")

NPB_BENCHMARKS = [
    {"binary": "is.C.x", "slug": "npb_is_C"},
    {"binary": "lu.B.x", "slug": "npb_lu_B"},
    {"binary": "sp.B.x", "slug": "npb_sp_B"},
    {"binary": "ua.B.x", "slug": "npb_ua_B"},
]

NPB_BENCHMARKS_SMALL = [
    {"binary": "is.W.x", "slug": "npb_is_W"},
    {"binary": "lu.W.x", "slug": "npb_lu_W"},
    {"binary": "sp.W.x", "slug": "npb_sp_W"},
    {"binary": "ua.W.x", "slug": "npb_ua_W"},
]

SPEC_BENCHMARKS = [
    {
        "suite": "531.deepsjeng_r",
        "slug": "spec_531_deepsjeng_r",
        "args": ["ref.txt"],
    },
    {
        "suite": "538.imagick_r",
        "slug": "spec_538_imagick_r",
        # imagevalidate_538_base sorts first alphabetically; name the real binary explicitly.
        "binary_name": "imagick_r_base.*",
        "args": [
            "-limit", "disk", "0",
            "refrate_input.tga",
            "-edge", "41",
            "-resample", "181%",
            "-emboss", "31",
            "-colorspace", "YUV",
            "-mean-shift", "19x19+15%",
            "-resize", "30%",
            "refrate_output.tga",
        ],
    },
]


def source_shrc(spec_root: Path) -> dict[str, str]:
    result = subprocess.run(
        ["bash", "-c", f"cd {spec_root} && source ./shrc && env"],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.partition("=")[::2] for line in result.stdout.splitlines() if "=" in line)


def spec_ensure_built(bench: str, spec_root: Path, env: dict, binary_name: str = "*_base.*") -> Path:
    exe_dir = spec_root / "benchspec/CPU" / bench / "exe"
    binaries = sorted(exe_dir.glob(binary_name)) if exe_dir.exists() else []
    if not binaries:
        try:
            subprocess.run(
                [
                    spec_root / "bin/runcpu",
                    "--action=build",
                    "--size=refrate",
                    bench,
                ],
                env=env,
                cwd=spec_root,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(
                f"ERROR: failed to build {bench}.\n"
                f"Run manually: source {spec_root}/shrc && "
                f"runcpu --action=build {bench}",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        binaries = sorted(exe_dir.glob(binary_name))
    if not binaries:
        print(
            f"ERROR: no binary in {exe_dir} after build.\n"
            f"Run manually: source {spec_root}/shrc && "
            f"runcpu --action=build {bench}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return binaries[0]


def spec_run_dir(bench: str, spec_root: Path, env: dict) -> Path:
    run_base = spec_root / "benchspec/CPU" / bench / "run"
    dirs = sorted(run_base.glob("run_base_refrate_*")) if run_base.exists() else []
    if not dirs:
        try:
            subprocess.run(
                [
                    spec_root / "bin/runcpu",
                    "--action=setup",
                    "--size=refrate",
                    bench,
                ],
                env=env,
                cwd=spec_root,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(
                f"ERROR: failed to set up run dir for {bench}.\n"
                f"Run manually: source {spec_root}/shrc && "
                f"runcpu --action=setup {bench}",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        dirs = sorted(run_base.glob("run_base_refrate_*"))
    return dirs[-1]


def run_sweep(
    saccade: Path,
    binary: Path,
    args: list[str],
    cwd: Path,
    out_h5: Path,
    trace: Path,
    library: Path | None,
    dry_run: bool,
) -> None:
    cmd = [
        str(saccade),
        "sweep",
        "--quiet",
        "--matrix",
        str(out_h5),
        "--trace",
        str(trace),
    ]
    if library:
        cmd += ["--library", str(library)]
    cmd += ["--", str(binary)] + args
    if dry_run:
        print(f"  $ {' '.join(cmd)}")
        return
    subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def merge_into(src_h5: Path, dst_h5: Path, group: str, replace: bool = False) -> None:
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "a") as dst:
        if group in dst:
            if replace:
                del dst[group]
            else:
                raise ValueError(f"Group '{group}' already exists in {dst_h5}. Use --replace to overwrite.")
        src.copy("/", dst, name=group)


def build_benchmark_list(
    run_npb: bool, run_spec: bool, spec_root: Path, small: bool
) -> list[dict]:
    benchmarks = []

    npb_list = NPB_BENCHMARKS_SMALL if small else NPB_BENCHMARKS

    if run_npb:
        for b in npb_list:
            benchmarks.append(
                {
                    "slug": b["slug"],
                    "binary": NPB_DIR / b["binary"],
                    "args": [],
                    "cwd": NPB_DIR,
                    "kind": "npb",
                }
            )

    if run_spec:
        env = source_shrc(spec_root)
        for b in SPEC_BENCHMARKS:
            binary = spec_ensure_built(b["suite"], spec_root, env, b.get("binary_name", "*_base.*"))
            cwd = spec_run_dir(b["suite"], spec_root, env)
            benchmarks.append(
                {
                    "slug": b["slug"],
                    "binary": binary,
                    "args": b["args"],
                    "cwd": cwd,
                    "kind": "spec",
                }
            )

    return benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect saccade ground truth evaluation sweep data across NPB and SPEC benchmarks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sweep_data_eval.h5"),
        help="Combined HDF5 output file (default: ./sweep_data_eval.h5)",
    )
    parser.add_argument(
        "--saccade",
        type=Path,
        default=Path("../target/release/saccade"),
        help="saccade binary (default: ./target/release/saccade)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("../event_lib.json"),
        help="Event library JSON (optional)",
    )
    parser.add_argument(
        "--npb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run NPB benchmarks (default: yes)",
    )
    parser.add_argument(
        "--spec",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run SPEC benchmarks (default: yes)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Run only NPB at class W (skip SPEC); for quick smoke tests",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="SLUG",
        default=None,
        help="Run only benchmarks with these slugs (e.g. spec_531_deepsjeng_r spec_538_imagick_r)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite existing groups in the HDF5 output file instead of erroring",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    if args.small:
        args.spec = False

    if not args.npb and not args.spec and not args.only:
        parser.error("At least one of --npb or --spec must be enabled.")

    args.saccade = args.saccade.resolve()
    args.output = args.output.resolve()
    if args.library:
        args.library = args.library.resolve()
    if not args.saccade.exists():
        parser.error(f"saccade binary not found: {args.saccade}")

    benchmarks = build_benchmark_list(args.npb, args.spec, SPEC_ROOT, args.small)

    if args.only:
        known = {b["slug"] for b in benchmarks}
        unknown = set(args.only) - known
        if unknown:
            parser.error(f"Unknown slug(s): {', '.join(sorted(unknown))}. Known: {', '.join(sorted(known))}")
        benchmarks = [b for b in benchmarks if b["slug"] in set(args.only)]

    trace_dir = args.output.parent / f"{args.output.stem}_traces"

    if args.dry_run:
        print(f"Would write combined output to: {args.output}")
        print(f"Would write per-benchmark traces to: {trace_dir}/")
        with tempfile.TemporaryDirectory() as tmp:
            for b in benchmarks:
                tmp_h5 = Path(tmp) / f"{b['slug']}.h5"
                trace = trace_dir / f"{b['slug']}.perfetto"
                run_sweep(
                    args.saccade,
                    b["binary"],
                    b["args"],
                    b["cwd"],
                    tmp_h5,
                    trace,
                    args.library,
                    dry_run=True,
                )
        return

    trace_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        bar = tqdm(benchmarks, desc="benchmarks", unit="bench")
        for b in bar:
            bar.set_postfix_str(b["slug"])
            tmp_h5 = Path(tmp) / f"{b['slug']}.h5"
            trace = trace_dir / f"{b['slug']}.perfetto"
            run_sweep(
                args.saccade,
                b["binary"],
                b["args"],
                b["cwd"],
                tmp_h5,
                trace,
                args.library,
                dry_run=False,
            )
            merge_into(tmp_h5, args.output, b["slug"], replace=args.replace)
            tmp_h5.unlink()

    print(f"Done. Output: {args.output}")
    print(f"Traces: {trace_dir}/")


if __name__ == "__main__":
    main()
