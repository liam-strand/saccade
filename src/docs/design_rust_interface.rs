//! # Rust Interface Design (Oculomotor)
//!
//! This document outlines how the Rust `Oculomotor` component drives the
//! sampling pipeline via a pluggable `CounterBackend`.
//!
//! ## 1. CounterBackend Trait
//!
//! The backend abstraction decouples Oculomotor from any specific data source.
//!
//! ```ignore
//! pub trait CounterBackend {
//!     fn poll_observations(&mut self) -> Vec<Observation>;
//!     fn update_counters(
//!         &mut self,
//!         old_set: &[EventId],
//!         new_set: &[EventId],
//!     ) -> Result<(), Box<dyn std::error::Error>>;
//! }
//! ```
//!
//! Two implementations:
//! - **`HardwareBackend`**: Owns the BPF skeleton, ring buffer, perf timer
//!   events, and `HardwareCounters`. Used by the `run` subcommand.
//! - **`VirtualBackend`**: Generates synthetic observations from golden rates
//!   (Normal distribution via `rand_distr`). Used by the `simulate` subcommand.
//!
//! ## 2. Oculomotor (Orchestrator)
//!
//! `Oculomotor` is backend-agnostic. It owns the scheduler, VCS, and active set.
//!
//! ```ignore
//! pub struct Oculomotor {
//!     backend: Box<dyn CounterBackend>,
//!     scheduler: Box<dyn Scheduler>,
//!     active_set: Vec<EventId>,
//!     _logger: Option<Logger>,
//!     vcs: VirtualCounterState,
//!     last_step_ns: u64,
//! }
//! ```
//!
//! ## 3. Hardware Counter Management
//!
//! The `HardwareCounters` struct manages the `perf_event` file descriptors and BPF map updates.
//! It is owned by `HardwareBackend`, not by `Oculomotor` directly.
//!
//! ### The `update_slot` method
//! When `HardwareBackend::update_counters` detects a slot change:
//!
//! 1.  **World-stop**: Set `tracking = false` in BPF globals; spin until all
//!     CPUs set their `stopped[cpu]` flag.
//! 2.  **Disable**: Call `disable()` on the old `perf_event` FD for the slot
//!     on each CPU (if one exists).
//! 3.  **Create & Enable**: Open a new `perf_event` FD per CPU for the
//!     requested event and immediately call `enable()`.
//! 4.  **Map Update**: For each CPU `c` and slot `i`, update BPF map
//!     `bpf_maps[i]` at key `c` with the new FD.
//! 5.  **Resume**: Set `tracking = true` in BPF globals.
//!
//! ## 4. Main Loop (`step()`)
//!
//! Each quantum, `Oculomotor::step()` executes:
//!
//! 1.  `backend.poll_observations()` — get aggregated per-event observations.
//! 2.  For each observed event: compute rate, call `vcs.measurement_update()`.
//! 3.  For each unobserved event: call `vcs.time_update()` (grow uncertainty).
//! 4.  `scheduler.next_step(&vcs)` — get the next `ScheduleDecision`.
//! 5.  `backend.update_counters(old_set, new_set)` — apply counter changes.
//!
//! ## Summary of Flow
//!
//! 1.  **User**: `saccade run -- <target>` or `saccade simulate --golden <rates>`
//! 2.  **Main**: Creates the appropriate backend (`HardwareBackend` or
//!     `VirtualBackend`) and passes it to `Oculomotor::new()`.
//! 3.  **Main Loop**: Calls `oculomotor.step()` each quantum until the target
//!     exits (run) or the step count is reached (simulate).
