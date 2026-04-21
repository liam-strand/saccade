use crate::event::EventId;
use crate::quantum::Quantum;
use crate::state::StateEstimator;

pub mod csv;
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
}
