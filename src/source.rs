//! Trait definition for sample sources and their hardware/virtual implementations.

use crate::event::EventId;
use crate::sample::RawSample;

pub mod hardware;
pub mod virtual_source;

/// Timing breakdown of a single `apply_schedule` call.
///
/// `swap_ns` (the wall-clock total) is measured by the caller; this struct
/// splits the hardware work into the stop-the-world quiesce spin-wait versus
/// the actual counter reconfiguration, and records how many slots changed so
/// no-op swaps (`slots_changed == 0`) can be distinguished downstream.
#[derive(Debug, Default, Clone, Copy)]
pub struct SwapStats {
    /// Time spent in `stop_counters` waiting for every CPU to acknowledge the
    /// quiesce, summed across each changed slot. Dominated by CPU round-up
    /// latency, not counter work.
    pub quiesce_ns: u64,
    /// Time spent opening fresh `perf_event` FDs and updating the BPF maps,
    /// summed across each changed slot. The true counter-reconfiguration cost.
    pub reconfig_ns: u64,
    /// Number of counter slots actually reconfigured this call.
    pub slots_changed: usize,
}

/// Abstraction over where performance counter samples come from.
///
/// Implementations return raw `RawSample` values (count + duration); rate
/// computation happens downstream in `Quantum::aggregates()`.
pub trait SampleSource {
    /// Collect all raw samples since the last call.
    ///
    /// Returns `(samples, elapsed_ns)` where `elapsed_ns` is the wall-clock
    /// time covered by this collection window.
    fn collect(&mut self) -> (Vec<RawSample>, u64);

    /// Switch which hardware events are being monitored.
    ///
    /// Called with the old and new active sets so the implementation can
    /// diff and only reconfigure changed slots.
    fn apply_schedule(
        &mut self,
        old_set: &[EventId],
        new_set: &[EventId],
    ) -> Result<SwapStats, Box<dyn std::error::Error>>;

    /// Number of hardware counter slots available simultaneously.
    /// `4` for eBPF hardware sources; configurable for virtual sources.
    fn num_slots(&self) -> usize;
}
