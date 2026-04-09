use crate::event::EventId;
use crate::sample::RawSample;

pub mod hardware;
pub mod virtual_source;

/// Abstraction over where performance counter samples come from.
///
/// Replaces `CounterBackend`. Key difference: returns raw `RawSample` values
/// (count + duration), not pre-aggregated `Observation` values.
/// Rate computation happens downstream in `Quantum::aggregates()`.
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
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Number of hardware counter slots available simultaneously.
    /// `4` for eBPF hardware sources; configurable for virtual sources.
    fn num_slots(&self) -> usize;
}
