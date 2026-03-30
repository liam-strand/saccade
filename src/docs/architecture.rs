//! # SACCADE ARCHITECTURE & IMPLEMENTATION NOTES
//! ## SYSTEM COMPONENTS
//!
//! The system implements a strict separation of concerns between mechanism (eBPF)
//! and policy (Rust).
//!
//! | LAYER                | COMPONENT  | TECHNOLOGY       | //! RESPONSIBILITY                                                                                     |
//! | :------------------- | :--------- | :--------------- | //! :------------------------------------------------------------------------------------------------- |
//! | **L4: Intelligence** | Scheduler  | Rust (pluggable trait) | Policy Layer. Determines counter selection based on pluggable policy. ML-steered scheduling is a planned future direction (technology TBD). |
//! | **L3: Control**      | Oculomotor | Rust + libbpf-rs | User Agent. Backend-agnostic orchestrator: aggregates observations, updates VirtualCounterState, executes scheduler policy. Delegates hardware details to a pluggable `CounterBackend` (`HardwareBackend` for real eBPF/perf, `VirtualBackend` for simulation). |
//! | **L2: Data**         | Retina     | eBPF (C)         | Sampling Layer. Implements Gated Sampling via sched_switch and //! perf_event.                         |
//! | **L1: Hardware**     | PMU        | Linux Perf       | Hardware Layer. Physical counters managed via standard //! perf_event_open.                            |
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
//!   state by recording the timestamp (in nanoseconds) when the task was
//!   scheduled in. This acts as both a gate and a reference for delta time.
//!
//! ### Trigger Logic
//!
//! 1. Context Switch Hook (`sched_switch`)
//!    * Switch-IN (Target):
//!      - Action: Set timestamp in Start Map.
//!      - Effect: Enables timer-based sampling with delta reference.
//!
//!    * Switch-OUT (Target):
//!      - Action: FLUSH (Record final sample) -> Delete entry in Start Map.
//!      - Effect: Captures execution tail; disables timer overhead.
//!
//! 2. Timer Hook (`perf_event`)
//!    * Frequency: High (15,000 Hz software CPU_CLOCK, one timer per CPU).
//!    * Action: Check Start Map.
//!      - If not present: Exit immediately.
//!      - If present: Record Intermediate Sample, update timestamp.
//!
//! ### Sequence Flow
//!
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
//! 2. Logical Groups:
//!    * The Scheduler returns `ScheduleDecision` containing a `Vec<EventId>`.
//!    * `Oculomotor` passes the old and new active sets to
//!      `backend.update_counters()`. `HardwareBackend` diffs the sets and calls
//!      `update_slot` only for slots whose event changed.
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
//!      6. Set `tracking = true` to resume sampling.
//!
//! ### SCHEDULER INTERFACE
//!
//! The scheduling logic is decoupled via a Rust Trait to allow swapping between
//! baseline and ML strategies.
//!
//! #### Trait Definition
//!
//! ```ignore
//! pub trait Scheduler {
//!     fn init(&mut self, all_events: Vec<EventId>);
//!     fn next_step(&mut self, state: &VirtualCounterState) -> ScheduleDecision;
//! }
//! ```
//!
//! `next_step` receives the current `VirtualCounterState`, which provides
//! per-counter rate estimates (EMA) and uncertainty values. This allows
//! intelligent schedulers to prioritize counters with high uncertainty or
//! interesting rate changes.
//!
//! * Round-Robin Scheduler:
//!   - Logic: Deterministic rotation through defined groups.
//!   - Use Case: Baseline profiling, data collection for training.
//!
//! * Random Scheduler:
//!   - Logic: Picks 4 events at random each step.
//!   - Use Case: Comparison baseline.
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
//! Hardware counters are monotonic. Userspace must derive rates based on trigger
//! type:
//!
//! * Intermediate Sample:
//!   $\Delta = V_{t} - V_{t-1}$
//!
//! * Flush Sample:
//!   $\Delta = V_{t} - V_{t-1}$
//!
//! * Switch-IN:
//!   $V_{t} = V_{t-1}$
//!   (Re-baseline; no data emitted).
//!
