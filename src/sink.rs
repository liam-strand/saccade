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
    fn emit(
        &mut self,
        quantum: &Quantum,
        estimator: &dyn StateEstimator,
        active_set: &[EventId],
    ) -> std::io::Result<()>;

    fn finish(&mut self) -> std::io::Result<()>;

    /// Called once before each sweep batch begins. Sinks that need batch-identity
    /// metadata (e.g. MatrixSink) override this; all others use this no-op default.
    fn begin_batch(&mut self, _batch_id: u32, _events: &[EventId]) {}
}

pub fn finish_sinks(sinks: &mut [Box<dyn OutputSink>]) {
    for sink in sinks.iter_mut() {
        let _ = sink.finish();
    }
}
