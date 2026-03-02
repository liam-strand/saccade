//! # SACCADE ARCHITECTURE & IMPLEMENTATION NOTES
//! ## SYSTEM COMPONENTS
//!
//! The system implements a strict separation of concerns between mechanism (eBPF)
//! and policy (Rust).
//!
//! | LAYER                | COMPONENT  | TECHNOLOGY       | //! RESPONSIBILITY                                                                                     |
//! | :------------------- | :--------- | :--------------- | //! :------------------------------------------------------------------------------------------------- |
//! | **L4: Intelligence** | Scheduler  | ONNX / Torch     | Policy Layer. Determines counter selection based on information //! gain.                              |
//! | **L3: Control**      | Oculomotor | Rust + libbpf-rs | User Agent. Manages FD lifecycle ("Hot Pool"), aggregates samples, executes policy, //! handles ioctl. |
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
//!    * Frequency: High (e.g., 50–100Hz).
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
//! Hardware counter reconfiguration must occur within microseconds. The system
//! avoids close/open syscalls during runtime using a "Hot Pool" strategy.
//!
//! ### Implementation Specifications
//!
//! 1. Initialization:
//!    * Open perf_event_open file descriptors (FDs) for ALL cataloged events
//!      at startup.
//!    * Initial state: disabled=1, pinned=0.
//!
//! 2. Logical Groups:
//!    * The Scheduler maintains "Active Groups" as lists of FDs, not
//!      kernel-side Perf Groups.
//!
//! 3. Actuation Routine:
//!    * To switch active sets:
//!      1. Disable and close existing `perf_event` file descriptors for the slot.
//!      2. Open new `perf_event` FDs for each CPU.
//!      3. `bpf_map_update_elem` on the specific `PERF_EVENT_ARRAY` for the slot (e.g., `counter0`) using `cpu` indexing.
//!      4. `ioctl(ENABLE)` on new FDs.
//!
//! ### SCHEDULER INTERFACE
//!
//! The scheduling logic is decoupled via a Rust Trait to allow swapping between
//! baseline and ML strategies.
//!
//! #### Trait Definition
//!
//! The scheduler must accept an Observation (aggregated rates) and return a
//! ScheduleDecision.
//!
//! * Round-Robin Scheduler:
//!   - Logic: Deterministic rotation through defined groups.
//!   - Use Case: Baseline profiling, data collection for training.
//!
//! * RL Scheduler:
//!   - Logic: ONNX model inference.
//!   - Input: Vectorized event rates + phase embedding.
//!   - Output: Optimal counter set ID.
//!
//! ## DATA HANDLING
//!
//! ### Event Catalog (event_lib.json)
//!
//! The mapping between logical ML features and hardware config values must be
//! decoupled.
//! ```json
//! [
//!   { "id": 0, "name": "instructions", "config": "0x0001" },
//!   { "id": 1, "name": "l3_miss_skylake", "config": "0x0151" }
//! ]
//! ```
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
