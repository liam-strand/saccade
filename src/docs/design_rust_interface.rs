//! # Rust Interface Design
//!
//! This document describes how the Rust orchestrator (`Profiler`) drives the
//! sampling pipeline via pluggable source, scheduler, estimator, and sink traits.
//!
//! ## 1. SampleSource Trait
//!
//! `SampleSource` decouples `Profiler` from any specific data origin.
//!
//! ```ignore
//! pub trait SampleSource {
//!     fn collect(&mut self) -> (Vec<RawSample>, u64); // (samples, elapsed_ns)
//!     fn apply_schedule(
//!         &mut self,
//!         old_set: &[EventId],
//!         new_set: &[EventId],
//!     ) -> Result<(), Box<dyn std::error::Error>>;
//!     fn num_slots(&self) -> usize;
//! }
//! ```
//!
//! Two implementations:
//! - **`HardwareSampleSource`**: Owns the BPF skeleton, ring buffer, perf timer
//!   events, and `HardwareCounters`. Used by the `run` and `sweep` subcommands.
//! - **`VirtualSampleSource`**: Generates synthetic `RawSample` values from
//!   `TimeVaryingRates` profiles (loaded from sweep Perfetto traces) using
//!   Gaussian noise via `rand_distr`. Used by the `simulate` subcommand.
//!
//! ## 2. Profiler (Orchestrator)
//!
//! `Profiler` is source-agnostic. It owns the source, scheduler, state estimator,
//! output sinks, and current active set.
//!
//! ```ignore
//! pub struct Profiler<'s> {
//!     source: Box<dyn SampleSource>,
//!     scheduler: Box<dyn Scheduler>,
//!     sinks: &'s mut [Box<dyn OutputSink>],
//!     estimator: Box<dyn StateEstimator>,
//!     active_set: Vec<EventId>,
//!     current_time_ns: u64,
//! }
//! ```
//!
//! Use `ProfilerBuilder` to assemble a `Profiler`; all four components (source,
//! scheduler, estimator, sinks) are required before calling `build()`.
//!
//! ## 3. Hardware Counter Management
//!
//! `HardwareCounters` manages `perf_event` file descriptors and BPF map updates.
//! It is owned by `HardwareSampleSource`, not by `Profiler`.
//!
//! ### The `update_slot` method
//! When `HardwareSampleSource::apply_schedule` detects a slot change:
//!
//! 1.  **World-stop**: Set `tracking = false` in BPF globals; spin until all
//!     CPUs set their `stopped[cpu]` flag.
//! 2.  **Disable**: Call `disable()` on the old `perf_event` FD for the slot
//!     on each CPU (if one exists).
//! 3.  **Create & Enable**: Open a new `perf_event` FD per CPU for the
//!     requested event and immediately call `enable()`.
//! 4.  **Map Update**: For each CPU `c` and slot `i`, update BPF map
//!     `bpf_maps[i]` at key `c` with the new FD.
//! 5.  **Resume**: Set `tracking = true` in BPF globals. Each CPU then emits a
//!     `SAMPLE_TYPE_RESUME` marker so userspace can re-anchor per-(cpu, slot)
//!     counter baselines.
//!
//! ## 4. OutputSink Trait
//!
//! Each `OutputSink` receives the completed `Quantum` and current estimator state
//! once per profiler step.
//!
//! ```ignore
//! pub trait OutputSink {
//!     fn emit(&mut self, quantum: &Quantum, estimator: &dyn StateEstimator,
//!             active_set: &[EventId]) -> std::io::Result<()>;
//!     fn finish(&mut self) -> std::io::Result<()>;
//!     fn begin_batch(&mut self, batch_id: u32, events: &[EventId]) {} // no-op default
//! }
//! ```
//!
//! Provided sinks: `CsvSink`, `MatrixSink`, `PerfettoSink`, `NullSink`.
//!
//! ## 5. Main Loop (`Profiler::step()`)
//!
//! Each quantum, `Profiler::step()` executes:
//!
//! 1.  `source.collect()` — drain the ring buffer; returns `(Vec<RawSample>, elapsed_ns)`.
//! 2.  Build a `Quantum` from the raw samples and timing metadata. `Quantum`
//!     lazily computes per-event and per-(tid, event) `EventAggregate` values
//!     (mean rate, stddev, count) using Welford's online algorithm.
//! 3.  **Estimator update**: for each `(tid, event)` observed this quantum call
//!     `estimator.measurement_update()`; for every known `(tid, event)` pair
//!     that was *not* observed call `estimator.time_update()` (grow uncertainty).
//!     Then call `estimator.quantum_step()` once per active tid to apply
//!     cross-event (off-diagonal) process noise.
//! 4.  `scheduler.next_step(&quantum, estimator)` — returns a `ScheduleDecision`
//!     containing the next `active_events` and an optional quantum `duration`.
//! 5.  `source.apply_schedule(old_set, new_set)` — reconfigure hardware counters.
//! 6.  `sink.emit(&quantum, estimator, &active_set)` for each registered sink.
//!
//! ## Summary of Flow
//!
//! 1.  **User**: `saccade run -- <target>` or `saccade simulate`
//! 2.  **Main**: Assembles a `Profiler` via `ProfilerBuilder`, wiring up the
//!     appropriate `SampleSource` (`HardwareSampleSource` or `VirtualSampleSource`),
//!     a `Scheduler`, a `StateEstimator`, and one or more `OutputSink`s.
//! 3.  **Main Loop**: Calls `profiler.step()` each quantum until the target
//!     exits (`run`) or the step count is reached (`simulate`).
