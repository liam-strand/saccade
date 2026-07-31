# Saccade

An eBPF-based Linux performance profiler that intelligently rotates hardware counter slots
across time quanta to estimate rates for **more counters than the PMU can measure
simultaneously**.

A modern x86 core exposes hundreds of hardware performance events but only a handful of
physical counter slots — six, in saccade's case. `perf stat` handles this with round-robin
multiplexing and linear scaling: each event gets a fraction of the wall clock, and its count
is extrapolated. That is a fixed policy with no notion of how stale a given measurement is.

Saccade separates the two halves of that problem:

- A **scheduler** decides which events occupy the six slots in each quantum. It is a Rust
  trait with several implementations — round-robin, random, uncertainty-driven,
  rate-of-change, and a family of LLM-driven policies that can be steered with a
  natural-language hint.
- A **state estimator** maintains a running belief about *every* event's rate, including
  the ones not currently being measured, along with an uncertainty for each. Options range
  from last-observation-carried-forward to a Kalman filter with cross-counter correlations
  learned offline.

The result is a time-varying rate estimate for the full event library, plus a calibrated
uncertainty signal, at a sampling overhead that stays low because the eBPF layer only fires
while the target is actually on-CPU.

The repository is both a working profiler and the artifact for an evaluation of these
policies; see [Evaluation harness](#evaluation-harness-python).

## How it works

Mechanism lives in eBPF, policy lives in Rust, and the two are strictly separated.

| Layer | Component | Technology | Responsibility |
| :--- | :--- | :--- | :--- |
| **L4: Intelligence** | Scheduler | Rust (pluggable trait) | Policy. Decides which events occupy the counter slots each quantum. |
| **L3: Control** | Profiler | Rust + libbpf-rs | Orchestrator. Collects `RawSample`s, builds a `Quantum`, updates the estimator, invokes the scheduler, applies the schedule, emits to sinks. Source-agnostic — real hardware or replayed simulation. |
| **L2: Data** | Sampler | eBPF (C) | Sampling. Gated sampling via `sched_switch` and `perf_event`. |
| **L1: Hardware** | PMU | Linux perf | Physical counters via `perf_event_open`. |

### Gated sampling

Rather than sampling on a free-running timer, the eBPF layer gates on whether the target is
actually running (`src/bpf/sampler.bpf.c`):

- **Switch-in** (`tp_btf/sched_switch`): record a timestamp for the incoming thread in
  `start_map`. This arms the thread.
- **Timer tick** (`perf_event`, per-CPU software `CPU_CLOCK` at 15 kHz): if the running
  thread is not in `start_map`, return immediately. If it is, and at least
  `min_sample_interval_ns` has elapsed, emit an intermediate sample.
- **Switch-out**: emit a final flush sample covering the execution tail, then delete the
  `start_map` entry.

Timer overhead therefore disappears entirely while the target is off-CPU, and short-lived
threads still get their tail recorded instead of being lost between ticks.

### World-stop counter rotation

Reconfiguring a slot has to be atomic with respect to sampling, or userspace will compute
deltas across two different events. `HardwareCounters::update_slot`
(`src/hardware_counters.rs`) does:

1. Clear the `tracking` flag in BPF global state.
2. Spin until every active CPU has acknowledged by setting its `stopped[cpu]` flag.
3. Disable the old `perf_event` FD for that slot on each CPU.
4. Open and enable a new FD per CPU, then `bpf_map_update_elem` into the slot's
   `PERF_EVENT_ARRAY` keyed by CPU.
5. Set `tracking` back to true.

On resume each CPU emits a `SAMPLE_TYPE_RESUME` marker carrying absolute counter readings,
which `HardwareSampleSource` uses to re-anchor its per-`(cpu, slot)` delta baselines
(`src/source/hardware.rs`). The quiesce and reconfiguration costs are measured separately
and logged on every swap — see `q8` in the evaluation harness.

Full design notes live in the rustdoc modules under `src/docs/` (`architecture`,
`design_data_structures`, `design_rust_interface`), published to GitHub Pages by
`.github/workflows/docs.yml`.

## Requirements

**Kernel.** Linux 5.8 or newer (`BPF_MAP_TYPE_RINGBUF`). CO-RE handles the pre/post-5.14
`task_struct.state` → `__state` rename, so a single binary works across that boundary.

**Build dependencies.**

```bash
sudo apt install clang llvm libelf-dev zlib1g-dev libhdf5-dev
```

`perf` must also be on `PATH` — `saccade generate` shells out to `perf list --details`.

**Hardware.** Six counter slots (`MAX_COUNTERS` in `src/bpf/sampler.h`) and at most 256
CPUs. The checked-in `event_lib.json` and `config/expert.toml` are **AMD Zen–specific**:
events are encoded as raw `event`/`umask` pairs, and `saccade sweep` requires an event
named `ex_ret_instr` to use as its normalization anchor. On other microarchitectures,
regenerate the library with `saccade generate` and expect to adjust the anchor.

## Build

```bash
cargo build --release
```

`build.rs` compiles `src/bpf/sampler.bpf.c` into `src/bpf/sampler.skel.rs` via
`libbpf-cargo`, and generates wire-format bindings from `src/bpf/sampler.h` with `bindgen`.
Both are generated artifacts — `sampler.skel.rs` is gitignored and must never be edited by
hand. If it looks wrong, `cargo clean && cargo build`.

## Privileges

`run` and `sweep` load eBPF programs, call `perf_event_open`, and `ptrace` the target, so
they need root:

```bash
sudo ./target/release/saccade run -- ./my_program
```

Alternatively, grant the binary capabilities once (this is the `setcap` entry in
`Scripts.toml`):

```bash
sudo setcap cap_bpf,cap_perfmon,cap_sys_ptrace+ep ./target/release/saccade
```

`generate`, `simulate`, `evaluate`, and `cargo test` do not need root.

## Quick start

The typical research workflow is a four-step loop: measure ground truth, replay it under a
policy, and score the result.

```bash
# 1. Build an event library from `perf list` on this machine.
./target/release/saccade generate event_lib.json

# 2. Measure ground truth: run the target repeatedly, six fixed counters at a time,
#    until every event in the library has been covered once.
sudo ./target/release/saccade sweep \
    --library event_lib.json \
    --trace ground_truth.perfetto \
    --matrix rates.h5 \
    -- ./my_program

# 3. Replay those rates through a scheduler + estimator, without touching hardware.
./target/release/saccade simulate \
    --library event_lib.json \
    --rates-trace ground_truth.perfetto \
    --scheduler max_uncertainty \
    --estimator kalman \
    --trace estimated.perfetto

# 4. Score the estimate against ground truth.
./target/release/saccade evaluate \
    --ground-truth ground_truth.perfetto \
    --estimated estimated.perfetto
```

For live profiling with real counter rotation, skip straight to `run`:

```bash
sudo ./target/release/saccade run \
    --library event_lib.json \
    --scheduler round_robin \
    --estimator ema \
    --trace trace.perfetto \
    -- ./my_program
```

Open any `.perfetto` output at [ui.perfetto.dev](https://ui.perfetto.dev).

## Commands

Global flags apply to every subcommand: `-v/--verbose` for debug output, and `--config
<PATH>` to point at a TOML config (defaults to `saccade.toml` in the current directory if
present).

### `generate <OUTPUT>`

Parse `perf list --details` and write an event library JSON file.

```bash
./target/release/saccade generate event_lib.json
```

### `run -- <TARGET> [ARGS...]`

Profile a target process with live counter rotation. The target is spawned under
`PTRACE_TRACEME` so profiling starts at `exec`.

| Flag | Meaning |
| :--- | :--- |
| `-l, --library <PATH>` | Event library JSON. Falls back to running `perf list` if omitted. |
| `--scheduler <KIND>` | Rotation policy (see [Schedulers](#schedulers)). |
| `--estimator <KIND>` | State estimator (see [Estimators](#estimators)). |
| `--guidance <TEXT>` | Natural-language hint for LLM schedulers. |
| `--llm-model`, `--llm-base-url`, `--llm-api-key` | Override the `[llm]` config section. |
| `-q, --q-schedule <NS>` | Scheduler quantum — how often slots rotate. |
| `--q-sample <NS>` | Minimum interval between eBPF-emitted samples. |
| `--q-output <NS>` | Perfetto emission cadence (0 = every scheduling quantum). |
| `--trace <PATH>` | Perfetto output (default `trace.perfetto`). |
| `--csv <PATH>` | Raw per-sample CSV output. |

```bash
sudo ./target/release/saccade run -q 5000000 --csv samples.csv -- ./my_program arg1
```

### `sweep -- <TARGET> [ARGS...]`

Run the target repeatedly, each time with a different fixed batch of six counters (one
anchor plus five user events), until the whole library has been covered. This is how ground
truth is produced — every event gets full, unmultiplexed coverage, at the cost of running
the workload many times.

| Flag | Meaning |
| :--- | :--- |
| `-l, --library <PATH>` | Event library JSON. |
| `-q, --q-schedule <NS>` | Scheduler quantum. |
| `--q-sample <NS>` | Minimum interval between samples; also the HDF5 time-bin width. |
| `--trace <PATH>` | Perfetto trace with per-event time-varying rates (default `trace.perfetto`). |
| `--matrix <PATH>` | HDF5 matrix output, `N_events × T` per thread — the ML training format. |
| `--quiet` | Suppress the batch progress bar, for scripted use. |

### `simulate`

Replay the time-varying rates recorded by a sweep, driving a scheduler and estimator without
touching hardware. This makes policy comparison cheap and perfectly reproducible: the same
trace and seed give the same result every time.

| Flag | Meaning |
| :--- | :--- |
| `-l, --library <PATH>` | Event library JSON (**required**; no `perf` fallback here). |
| `-r, --rates-trace <PATH>` | Perfetto trace from `sweep --trace` (**required**). |
| `--scheduler`, `--estimator`, `--guidance` | Policy selection, as for `run`. |
| `--llm-model`, `--llm-base-url`, `--llm-api-key` | LLM overrides. |
| `-q, --q-schedule`, `--q-sample`, `--q-output` | Quanta, in nanoseconds. |
| `--noise-stddev <F>` | Gaussian noise on simulated rates (0 = none). |
| `--seed <N>` | RNG seed for reproducibility (omit for OS-random). |
| `--num-slots <N>` | Simulated hardware counter slots (default 6). |
| `--trace <PATH>`, `--csv <PATH>` | Outputs. |
| `--llm-latency-profile <PATH>` | JSON latency samples from `q7_llm_latency.py`, replayed in place of live LLM call latency. |
| `--batch <PATH>` | JSON list of combos to run in parallel. Each entry: `{scheduler, estimator, trace, seed?, guidance?, csv?}`. Overrides `--scheduler`/`--estimator`/`--trace`/`--seed`/`--guidance`. |
| `--jobs <N>` | Rayon threads for batch mode (0 = logical CPU count). |

Batch mode is how the evaluation sweeps a scheduler × estimator grid in one process:

```bash
./target/release/saccade simulate \
    -l event_lib.json -r ground_truth.perfetto \
    --batch combos.json --jobs 16
```

### `evaluate`

Compare a `simulate` output trace against a `sweep` ground-truth trace.

| Flag | Meaning |
| :--- | :--- |
| `--ground-truth <PATH>` | Trace from `sweep --trace`. |
| `--estimated <PATH>` | Trace from `simulate --trace`. |
| `--bin-ms <N>` | Time bin width in milliseconds (default 100, must be > 0). |
| `--json` | Emit JSON instead of a text table. |

## Schedulers

Selected with `--scheduler` or the `scheduler` key in TOML. Defined in `src/scheduler/`.

| Kind | Behavior |
| :--- | :--- |
| `round_robin` | *(default)* Slides a window of `num_slots` events across the library, advancing by `num_slots` each step. Deterministic full coverage. |
| `random` | Samples `num_slots` events uniformly at random, without replacement, each quantum. |
| `max_uncertainty` | Picks the events whose estimates are currently most uncertain. |
| `rate_of_change` | Prioritizes events with the highest non-linearity, using the Lim 2014 triangle-area cost. |
| `static_llm` | Queries an LLM once at startup for a cyclic schedule, then follows it. |
| `dynamic_llm` | Re-queries the LLM periodically (on a background thread) to adapt at runtime. |
| `weighted_round_robin_llm` | Round-robins according to per-counter weights assigned by the LLM. |
| `reasoning_static_llm` | `static_llm` with a free-form reasoning pass before schedule generation. |
| `reasoning_dynamic_llm` | `dynamic_llm` with a reasoning pass before each regeneration. |

LLM schedulers constrain the model's output with a JSON schema
(`{"steps": [{duration_ms, events}]}`, with the event list restricted to valid IDs and
bounded by `num_slots`) and retry with the malformed response fed back in. See
`src/scheduler/llm_common.rs`.

### Writing a new scheduler

The trait is the extension point:

```rust
pub trait Scheduler {
    fn init(&mut self, all_events: Vec<EventId>, num_slots: usize)
        -> Result<(), Box<dyn std::error::Error>>;
    fn next_step(&mut self, quantum: &Quantum, estimator: &dyn StateEstimator)
        -> ScheduleDecision;
}
```

`next_step` receives the current `Quantum` (raw samples plus lazily computed per-event
aggregates) and a read-only view of the estimator, which exposes per-`(tid, event)` rate
estimates and uncertainties. It returns a `ScheduleDecision` — an active event set no larger
than `num_slots`, and an optional quantum duration override. Schedulers never see file
descriptors; `HardwareCounters` owns all of that.

Three edits wire a new policy in:

1. Implement `Scheduler` in `src/scheduler/foo.rs` and add `pub mod foo;` to
   `src/scheduler.rs`.
2. Add a variant to `SchedulerKind` in `src/config.rs`. The `clap::ValueEnum` and serde
   derives use `rename_all = "snake_case"`, so the `--scheduler foo` flag and the TOML key
   come for free.
3. Add the matching `Display` arm and the `ResolvedConfig::build_scheduler` arm, also in
   `src/config.rs`.

The `scheduler_kind_display_round_trip` test asserts the `Display` strings stay in sync.

## Estimators

Selected with `--estimator` or the `estimator` key in TOML. Defined in `src/state/`. Every
estimator exposes a `CounterEstimate` per `(tid, event)`: `rate`, `rate_stddev`,
`uncertainty`, `last_updated_ns`, and `sample_count`.

| Kind | Behavior |
| :--- | :--- |
| `propagate` | *(default)* Last observation carried forward, unchanged, until the counter is re-sampled. Reports zero uncertainty. |
| `ema` | Exponential moving average, with uncertainty growing while a counter goes unobserved. Tuned via the `[ema]` config section (`alpha`, `uncertainty_growth_rate`). |
| `kalman` | Kalman filter with optional cross-counter correlation, so measuring one event updates beliefs about correlated events. Correlations are learned offline and loaded from JSON (`python/correlation.json`). |

`config/expert.toml` holds a Kalman configuration calibrated against sweep data across
seven benchmarks and 223 events, and documents how each parameter was derived.

## Configuration

Values resolve in order: **built-in defaults → TOML file → CLI flags**. The TOML file is
`saccade.toml` in the working directory, or whatever `--config` points at.

| Key | Default | Meaning |
| :--- | :--- | :--- |
| `scheduler` | `round_robin` | Rotation policy. |
| `estimator` | `propagate` | State estimator. |
| `q_schedule_ns` | `10_000_000` (10 ms) | How often the scheduler rotates counters. |
| `q_sample_ns` | `100_000` (100 µs) | Minimum interval between eBPF-emitted samples. |
| `q_output_ns` | `0` | Minimum interval between output flushes (0 = every quantum). |
| `noise_stddev` | `0.0` | Simulated measurement noise (simulation only). |
| `seed` | random | RNG seed. |

`saccade.toml` at the repo root is a fully commented reference covering these plus the
`[llm]`, `[ema]`, and `[kalman]` sections.

### LLM configuration

The LLM schedulers talk to any OpenAI-compatible chat-completions endpoint. **The
compiled-in default `base_url` is a Northwestern lab host
(`http://dubliner.cs.northwestern.edu:11434`)**, which will not be reachable elsewhere — set
your own before using an LLM scheduler.

```toml
[llm]
# Local Ollama
base_url = "http://localhost:11434"
model = "gemma4"

# ...or OpenRouter
# base_url = "https://openrouter.ai/api"
# api_key = "sk-or-..."

guidance = "I'm most interested in the memory behavior of this program"
```

Or per-invocation: `--llm-base-url`, `--llm-model`, `--llm-api-key`, `--guidance`.

## Output formats

### Perfetto trace (`--trace`)

A real `.perfetto-trace` binary (length-prefixed `TracePacket`s), openable in
[ui.perfetto.dev](https://ui.perfetto.dev). Each `(thread, event)` pair gets two counter
tracks, parented to a thread track:

- `{event_name}/rate` — estimated rate in events per nanosecond.
- `{event_name}/uncertainty` — estimator uncertainty in `[0, 1]`.

Emission is throttled by `q_output_ns`. These traces are also the input to `simulate
--rates-trace` and `evaluate`, read back via `perfetto::read_rate_timeseries`.

### CSV (`--csv`)

One row per raw sample — raw counts, not rates:

```
timestamp_ns,duration_ns,cpu_id,pid,tid,event_id,count,task
```

### HDF5 matrix (`sweep --matrix`)

The ML training format, written by `src/sink/matrix.rs`.

- Root attributes: `dt_ns` (bin width, equal to `q_sample_ns`) and `n_events`.
- `/event_names` — 1-D variable-length UTF-8, length `n_events`.
- `/batch_id` — 1-D `i32`, which sweep batch produced each event row (`-1` = never sampled).
- `/thread_<tid>/rates` — `f32[n_events × n_timesteps]`, mean rate per bin in events/ns,
  `NaN` where a bin had no samples. Group attributes: `task_name`, `tgid`, `n_timesteps`.

Thread IDs are synthetic, assigned by `(task_name, ordinal within that name)` so they are
stable across the many runs a sweep performs.

Because each batch is a separate execution of the workload, raw counts are not directly
comparable across batches. The anchor event `ex_ret_instr` is present in every batch and
stored as plain events/ns; every other event is accumulated as events *per instruction* and
rescaled by the global instruction rate at the end, which makes rows from different batches
commensurable.

### Event library (`generate`)

```json
{
  "events": [
    { "name": "instructions", "desc": "Retired instructions", "event": 192, "umask": 0 },
    { "name": "l3_miss_skylake", "desc": "L3 cache miss", "event": 46, "umask": 65 }
  ]
}
```

`event` and `umask` are decimal integers passed straight to `perf_event_open`. An `EventId`
is simply the positional index into this list, assigned at load time by `EventRegistry`.

### Evaluation metrics (`evaluate`)

A text table by default; `--json` gives:

```json
{
  "ground_truth": "...", "estimated": "...", "bin_width_ms": 100,
  "per_event": [
    { "event": "...", "tid": 0, "nrmse": 0.0, "coverage": 0.0,
      "mean_gt_rate_events_per_ns": 0.0, "calibration": 0.0, "gt_cv": 0.0 }
  ],
  "mean_nrmse": 0.0, "events_with_zero_coverage": 0,
  "mean_coverage": 0.0, "mean_calibration": 0.0
}
```

- **nRMSE** — normalized RMSE between estimated and ground-truth rates, scored only over
  ground-truth bins at or after the estimator's first observation of that event. This avoids
  penalizing a policy for the latency before it first reaches an event.
- **coverage** — fraction of *all* ground-truth bins for which an estimate exists.
- **calibration** — fraction of scored bins where the ground-truth rate falls inside
  `[est × (1 − unc), est × (1 + unc)]`, i.e. whether the reported uncertainty is honest.

## Evaluation harness (`python/`)

The `python/` directory holds the data-collection and analysis pipeline behind the project's
evaluation. It requires Python ≥ 3.14 and is managed with [uv](https://docs.astral.sh/uv/).
Run everything from `python/`.

```bash
cd python
uv run python run_all.py                 # full collection + analysis
uv run python run_all.py --skip-collect  # re-plot from existing data
uv run python run_all.py --from q2       # resume collection at q2
uv run python run_all.py --steps q4,q5   # run only these collectors
uv run python run_all.py --dry-run       # echo commands, run nothing
```

`run_all.py` sequences the collectors in dependency order (`q1, q7, q2, q3, q6, q8, q4, q5`),
forwards outputs between them, then runs every analysis script. Figures land in
`python/results/`.

| Script | Question |
| :--- | :--- |
| `q1_overhead.py` | Wall-clock overhead imposed by saccade across a parameter grid, relative to an unprofiled baseline. |
| `q2_accuracy.py` | Accuracy of every scheduler × estimator combination across workload traces. The main grid. |
| `q3_kf_variants.py` | Kalman filter variants (naive / analytical / expert correlations) under estimator-independent schedulers. |
| `q4_llm_guidance.py` | Whether a natural-language guidance hint improves LLM-driven scheduling. |
| `q5_best_vs_baseline.py` | Best saccade config vs. baseline saccade vs. actual `perf stat`. |
| `q6_noise_floor.py` | Run-to-run sweep variability — the intrinsic noise floor for the accuracy experiments. |
| `q7_llm_latency.py` | LLM call latency distributions per scheduler call type; produces the profile replayed by `simulate --llm-latency-profile`. |
| `q8_swap_latency.py` | Cost of the counter-rotation path, split into quiesce (stop-the-world spin) and reconfiguration time. |

Plotting lives in `python/analysis/`, with shared styling in `analysis/plot_style.py`.
Supporting scripts include `collect.py`/`collect2.py` (sweep data across NPB and SPEC CPU2017
into HDF5), `correlation.py`/`expert_correlation.py` (derive the Kalman correlation
matrices), and `perf_to_perfetto.py`.

> **Reproducing outside the original environment:** `run_all.py` hard-codes benchmark paths
> under `/tank/yhe7443/benchmarks/NPB3.3.1/...` and expects pre-collected ground-truth traces
> in `python/sweep_data_eval_traces/`. Running it elsewhere means editing those constants at
> the top of the file and collecting your own sweep traces first.

### Synthetic workload

`src/bin/workload.rs` is a standalone benchmark binary with phases chosen to exercise
distinct counter profiles — useful when you have no benchmark suite handy:

```bash
cargo build --release
sudo ./target/release/saccade run -- ./target/release/workload phases.json
```

The JSON config is a list of phases, each with `duration_secs`, `threads`, and a `type` of
`cache_thrash`, `fp_heavy`, `branch_mispredict`, `tlb_thrash`, `mem_stream`, or `int_div`.

## Development

```bash
cargo test     # unit tests (event library parser); no root required
cargo clippy   # lint
cargo fmt      # format Rust
clang-format -i src/bpf/*.c src/bpf/*.h   # format the eBPF C
```

Conventions:

- Modules with submodules use the `foo.rs` + `foo/` layout, not `foo/mod.rs`.
- Never edit `src/bpf/sampler.skel.rs` — it is generated by `build.rs`. If it looks wrong,
  `cargo clean && cargo build`.

The only CI workflow is `.github/workflows/docs.yml`, which builds rustdoc on pushes to
`main` and deploys it to GitHub Pages.

## License

This project does not currently carry a top-level license file, and `Cargo.toml` declares no
`license` field. The eBPF source (`src/bpf/sampler.bpf.c`) is marked
`SPDX-License-Identifier: GPL-2.0`, as required for kernel-loaded BPF programs.
