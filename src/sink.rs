//! Trait and helpers for pluggable profiling output sinks.

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::state::StateEstimator;

pub mod csv;
pub mod matrix;
pub mod null;
pub mod perfetto;

/// Pluggable output consumer. Each sink receives the full `Quantum` (raw samples +
/// lazy aggregates) and current estimator state once per profiler step.
pub trait OutputSink {
    /// Consume one profiler quantum, writing whatever output the sink produces.
    fn emit(
        &mut self,
        quantum: &Quantum,
        estimator: &dyn StateEstimator,
        active_set: &[EventId],
    ) -> std::io::Result<()>;

    /// Flush and close the sink; called once after all quantums have been emitted.
    fn finish(&mut self) -> std::io::Result<()>;

    /// Called once before each sweep batch begins; sinks that need batch-identity
    /// metadata (e.g. `MatrixSink`) override this; all others use this no-op default.
    fn begin_batch(&mut self, _batch_id: u32, _events: &[EventId]) {}

    /// Configure instruction-count normalization for sweep output. Called once after
    /// all batches complete. `anchor_id` is the event whose count is the denominator;
    /// `global_ref_rate` (instructions/ns) is the multiplier used to convert
    /// events/instruction back to events/ns. Sinks that don't support normalization
    /// ignore this call via the default no-op.
    fn set_anchor(&mut self, _anchor_id: EventId, _global_ref_rate: f64) {}
}

/// Call `finish` on every sink, ignoring individual errors.
pub fn finish_sinks(sinks: &mut [Box<dyn OutputSink>]) {
    for sink in sinks.iter_mut() {
        let _ = sink.finish();
    }
}
