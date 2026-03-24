# CLAUDE.md

## Project Overview

Saccade is a Linux performance profiler that uses eBPF to intelligently sample hardware performance counters with minimal overhead. It spawns a target process, attaches eBPF programs to monitor context switches and timer events, and dynamically rotates which hardware counters are active based on a pluggable scheduler policy.

## Architecture

Four layers, from low-level to high-level:

1. **Hardware (PMU)** — CPU performance monitoring units provide raw counter values
2. **eBPF (Retina)** — Kernel-side C code (`src/bpf/sampler.bpf.c`) hooks `sched_switch` and `perf_event` timer to collect samples into a ring buffer
3. **Rust (Oculomotor)** — Userspace orchestrator manages eBPF lifecycle, hardware counter FDs, and ring buffer polling
4. **Scheduler** — Trait-based policy deciding which 4 counters to activate each quantum

## Languages

- **Rust** (2024 edition) — all userspace code
- **C** — eBPF kernel programs (`src/bpf/`)

## Build & Run

Requires Linux with eBPF support (5.8+ for ringbuf), `clang`/`llvm` for eBPF compilation, and `perf` installed.

```bash
cargo build                    # builds Rust + compiles eBPF via build.rs
sudo cargo run -- generate event_lib.json   # generate hardware event library
sudo cargo run -- run -- <target> [args...]  # profile a target program
```

The `build.rs` script uses `libbpf-cargo` SkeletonBuilder to compile `src/bpf/sampler.bpf.c` into `src/bpf/sampler.skel.rs` (gitignored, auto-generated).

All `cargo run` invocations require **sudo** (configured in `.cargo/config.toml`) because eBPF operations need root privileges.

## CLI

Two subcommands (defined in `src/cli.rs`):

- `generate <output>` — runs `perf list --details`, parses output with nom, writes JSON event library
- `run [--library <path>] [--quantum <ns>] -- <target> [args...]` — profiles a target program
  - `--library`: path to pre-generated event library JSON (otherwise generates on the fly)
  - `--quantum`: scheduler quantum in nanoseconds (default: 1,000,000 = 1ms)

## Source Structure

```
src/
├── main.rs              # Entry point: CLI parsing, process spawning, main loop
├── lib.rs               # Module declarations
├── cli.rs               # Clap command definitions
├── oculomotor.rs        # eBPF lifecycle, ring buffer polling, counter updates
├── hardware_counters.rs # Perf FD pool: open/enable/disable per-CPU counters
├── event_library.rs     # Nom parser for `perf list` output → Event structs
├── event_registry.rs    # EventId ↔ Event mapping
├── scheduler.rs         # Scheduler trait + ScheduleDecision struct
├── scheduler/
│   ├── random.rs        # Randomly picks 4 events each step (10ms duration)
│   ├── round_robin.rs   # Cycles through events 4 at a time
│   ├── distribution.rs  # (empty, placeholder)
│   └── test.rs          # Test scheduler with hardcoded event names
├── buffered_output.rs   # Threaded CSV logger with 8MB BufWriter + sync_channel
├── perf.rs              # Runs `perf list --details` as a subprocess
├── syscalls.rs          # Safe wrappers: ptrace, wait4, sched_setaffinity, etc.
├── docs/                # Architecture/design documentation as doc comments
│   ├── architecture.rs
│   ├── design_data_structures.rs
│   └── design_rust_interface.rs
└── bpf/
    ├── sampler.h        # Shared C header (saccade_sample struct, constants)
    ├── sampler.bpf.c    # eBPF programs (sched_switch hook + timer hook)
    └── sampler.skel.rs  # Auto-generated skeleton (gitignored)
```

## Key Constants

- `MAX_COUNTERS = 4` — simultaneous hardware counters per CPU
- `MAX_CPUS = 256` — maximum supported CPUs
- `TASK_COMM_LEN = 16` — Linux task name length
- Ring buffer size: 256 KB
- Logger buffer: 8 MB, channel capacity: 256,000 samples
- Default min sample interval: 1ms (1,000,000 ns)
- Timer sample frequency: 100,000 Hz

## Testing

```bash
cargo test           # runs unit tests (event_library parser tests)
cargo clippy         # lint
cargo fmt            # format
```

Unit tests are in `src/event_library.rs` (`#[cfg(test)]` module) testing the nom parser against sample `perf list` output.

Example/integration binaries are in `src/bin/` (`test_raw.rs`, `test_multicpu.rs`, `test_multiplex.rs`) and require sudo to run.

## Key Design Patterns

- **Scheduler trait** (`src/scheduler.rs`): `init(events)` + `next_step() -> ScheduleDecision`. Decision contains up to 4 `EventId`s and an optional duration.
- **Hot Pool** (`src/hardware_counters.rs`): Hardware counter FDs are opened per-slot-per-CPU. When the scheduler changes an event in a slot, only that slot's counters are replaced (disable old, build new, enable, update BPF map).
- **eBPF control channel**: Global variables in BSS/data sections (`target_pid`, `min_sample_interval_ns`, `active_counter_ids`) written from Rust via `skel.maps.bss_data` / `skel.maps.data_data`.
- **Gated sampling**: eBPF only records samples for tasks tracked in `start_map`. Context switch-in adds entry, switch-out flushes and removes. Timer hook emits intermediate samples if enough time has elapsed.
- **EventId** is a `u32` index into the `EventRegistry`'s event vector.

## Output

Samples are written to `saccade.csv` with columns:
```
timestamp_ns,duration_ns,pid,cpu_id,type,values_0,values_1,values_2,values_3,events_0,events_1,events_2,events_3,task
```

## Future Direction

The `Scheduler` trait is designed to eventually support ML-steered counter selection. The current schedulers (random, round-robin, test) serve as baselines and data-collection tools. A future ML scheduler would use sample data to make informed decisions about which counters to activate, but the approach (ONNX, PyTorch, etc.) is not yet decided. The trait interface may need to evolve to accept observation data when this work begins.
