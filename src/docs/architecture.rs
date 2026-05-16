//! # SACCADE ARCHITECTURE & IMPLEMENTATION NOTES
//! ## SYSTEM COMPONENTS
//!
//! The system implements a strict separation of concerns between mechanism (eBPF)
//! and policy (Rust).
//!
//! | LAYER                | COMPONENT  | TECHNOLOGY             | RESPONSIBILITY |
//! | :------------------- | :--------- | :--------------------- | :------------- |
//! | **L4: Intelligence** | Scheduler  | Rust (pluggable trait) | Policy Layer. Determines counter selection. Implementations range from round-robin baselines to LLM-driven adaptive schedulers. |
//! | **L3: Control**      | Profiler   | Rust + libbpf-rs       | Orchestrator. Source-agnostic: collects `RawSample`s, builds a `Quantum`, updates the `StateEstimator`, invokes the scheduler, applies the schedule, and emits to `OutputSink`s. Delegates hardware details to a pluggable `SampleSource` (`HardwareSampleSource` for eBPF/perf, `VirtualSampleSource` for simulation). |
//! | **L2: Data**         | Sampler    | eBPF (C)               | Sampling Layer. Implements Gated Sampling via `sched_switch` and `perf_event`. |
//! | **L1: Hardware**     | PMU        | Linux Perf             | Hardware Layer. Physical counters managed via `perf_event_open`. |
//!
//! ## SAMPLING LOGIC (EBPF)
//!
//! To balance resolution with overhead, the system uses a "Gated Sampling with
//! Flush-on-Eviction" strategy. This prevents data loss for short tasks and
//! eliminates overhead during target inactivity.
//!
//! ### State Management
//!
//! * Start Map: A BPF Hash Map shared between hooks tracks the target process
//!   state by recording the timestamp (in nanoseconds) when the task was last
//!   sampled or scheduled in. This acts as both a gate and a reference for
//!   computing `duration_ns`.
//!
//! * Global Flags: `tracking` (bool) enables/disables all sampling; `stopped[cpu]`
//!   (bool array) lets each CPU signal to userspace that it has halted, enabling
//!   the world-stop barrier used when reconfiguring hardware counters.
//!
//! ### Trigger Logic
//!
//! 1. Context Switch Hook (`sched_switch`)
//!    * Switch-IN (Target):
//!      - Action: Insert timestamp into Start Map for the incoming thread.
//!      - Effect: Enables timer-based sampling with delta reference.
//!
//!    * Switch-OUT (Target):
//!      - Action: FLUSH (record final sample with `delta = now − start`) →
//!        delete entry from Start Map.
//!      - Effect: Captures the execution tail; disables timer overhead for
//!        this thread while it is off-CPU.
//!
//! 2. Timer Hook (`perf_event`)
//!    * Frequency: 15,000 Hz software `CPU_CLOCK`, one perf event per CPU.
//!    * Action: Check Start Map for the currently running thread.
//!      - If not present: Exit immediately (thread is not being tracked).
//!      - If present but `delta < min_sample_interval_ns`: Exit (rate limiting).
//!      - Otherwise: Record `SAMPLE_TYPE_INTERMEDIATE` and update timestamp.
//!
//! 3. Resume Marker (`SAMPLE_TYPE_RESUME`)
//!    * Emitted when a CPU transitions from `stopped[cpu]=true` back to active.
//!    * Carries the current absolute counter readings so userspace can reset
//!      its per-(cpu, slot) baselines before computing deltas.
//!    * Produces no `RawSample`; consumed only by `HardwareSampleSource::wire_to_raw`.
//!
//! ### Sequence Flow
//!
//! ```mermaid
//! (Target Inactive - Start Map: Empty)
//! [OS Scheduler] -- Switch IN (Target) --> [eBPF]
//! [eBPF] -- Set Timestamp --> [Start Map]
//!
//!    LOOP: Timer Tick
//!    [eBPF] -- Check State --> [Start Map]
//!    [Start Map] -- Returns Timestamp --> [eBPF]
//!    [eBPF] -- Push Sample (Intermediate) --> [Userspace]
//!
//! [OS Scheduler] -- Switch OUT (Target) --> [eBPF]
//! [eBPF] -- Push Sample (Flush) --> [Userspace]
//! [eBPF] -- Delete Entry --> [Start Map]
//!
//! (Target Inactive - Start Map: Empty)
//!    LOOP: Timer Tick
//!    [eBPF] -- Check State --> [Start Map]
//!    [Start Map] -- Entry Not Found --> [eBPF]
//!    (Exit - No Ops)
//! ```
//!
//! ## RESOURCE MANAGEMENT (USERSPACE)
//!
//! Hardware counter slots are reconfigured on demand via a world-stop mechanism
//! that briefly pauses eBPF sampling to ensure consistent counter state.
//!
//! ### Implementation Specifications
//!
//! 1. Initialization:
//!    * `HardwareCounters` is created with empty slots (no FDs pre-allocated).
//!    * Counters are opened on demand when `update_slot` is first called.
//!
//! 2. Slot Management:
//!    * The Scheduler returns `ScheduleDecision` containing a `Vec<EventId>`.
//!    * `Profiler` passes the old and new active sets to
//!      `source.apply_schedule()`. `HardwareSampleSource` compares the sets
//!      positionally (slot-by-slot) and calls `HardwareCounters::update_slot`
//!      only for slots whose event changed.
//!    * `HardwareCounters` manages all perf event FDs; the Scheduler never
//!      sees or touches FDs directly.
//!
//! 3. Actuation Routine (world-stop):
//!    * To switch active sets:
//!      1. Set `tracking = false` in BPF global state, signalling eBPF hooks to
//!         stop sampling and set their per-CPU `stopped[cpu]` flag.
//!      2. Spin until all active CPUs report `stopped[cpu] == true`.
//!      3. Disable the old `perf_event` FD for the slot on each CPU.
//!      4. Open a new `perf_event` FD per CPU for the requested event and
//!         immediately enable it.
//!      5. `bpf_map_update_elem` on the slot's `PERF_EVENT_ARRAY` (e.g.,
//!         `counter0`) keyed by `cpu_id`.
//!      6. Set `tracking = true` to resume sampling. Each CPU then emits a
//!         `SAMPLE_TYPE_RESUME` marker so userspace can re-anchor baselines.
//!
//! ### SCHEDULER INTERFACE
//!
//! The scheduling logic is decoupled via a Rust trait.
//!
//! #### Trait Definition
//!
//! ```ignore
//! pub trait Scheduler {
//!     fn init(&mut self, all_events: Vec<EventId>, num_slots: usize)
//!         -> Result<(), Box<dyn std::error::Error>>;
//!     fn next_step(&mut self, quantum: &Quantum, estimator: &dyn StateEstimator)
//!         -> ScheduleDecision;
//! }
//! ```
//!
//! `next_step` receives the current `Quantum` (raw samples + lazy aggregates)
//! and a `StateEstimator` snapshot, which provides per-(tid, event) rate
//! estimates and uncertainty values. The returned active set must not exceed
//! `num_slots`.
//!
//! * Round-Robin Scheduler:
//!   - Logic: Activates a sliding window of `num_slots` events; the window
//!     advances by `num_slots` positions each step, wrapping around the full
//!     event list.
//!   - Use Case: Baseline profiling, deterministic coverage.
//!
//! * Fixed Scheduler:
//!   - Logic: Returns the same counter set every step; ignores the event universe
//!     passed to `init`.
//!   - Use Case: Used by the `sweep` command to hold a constant set of counters
//!     for an entire run, producing a ground-truth measurement.
//!
//! * Random Scheduler:
//!   - Logic: Samples `num_slots` events at random (without replacement) each step.
//!   - Use Case: Comparison baseline.
//!
//! * LLM Schedulers (`StaticLlmScheduler`, `DynamicLlmScheduler`,
//!   `WeightedRoundRobinLlmScheduler`):
//!   - Logic: Query an external LLM to generate a cyclic schedule, optionally
//!     refreshing it periodically based on the current estimator state.
//!   - Use Case: Adaptive, human-readable scheduling driven by model intuition.
//!
//! ## DATA HANDLING
//!
//! ### Event Catalog (event_lib.json)
//!
//! The mapping between logical ML features and hardware config values must be
//! decoupled.
//! ```json
//! {
//!   "events": [
//!     { "name": "instructions", "desc": "Retired instructions", "event": 192, "umask": 0 },
//!     { "name": "l3_miss_skylake", "desc": "L3 cache miss", "event": 46, "umask": 65 }
//!   ]
//! }
//! ```
//!
//! The `event` and `umask` fields are raw `u64` values passed directly to
//! `perf_event_open`. `EventId` is a positional index into this list, assigned
//! at load time by `EventRegistry`.
//!
//! ### Rate Calculation (Delta Math)
//!
//! Hardware counters hold **absolute** readings. `HardwareSampleSource` maintains
//! per-(cpu, slot) baselines and computes deltas before producing `RawSample`s.
//!
//! * Intermediate and Flush samples:
//!   $\Delta = V_{\text{abs}} - V_{\text{baseline}}$; baseline is then advanced
//!   to $V_{\text{abs}}$.
//!
//! * Resume marker (`SAMPLE_TYPE_RESUME`):
//!   Resets per-(cpu, slot) baselines to the current absolute counter values
//!   without emitting a `RawSample`. This re-anchors deltas after a world-stop
//!   reconfiguration.
//!
//! * Switch-IN:
//!   Records the current timestamp into `start_map` for the incoming thread.
//!   No counter read or delta computation occurs at this point.
//!
