use crate::event::EventId;
use crate::quantum::Quantum;
use crate::sink::OutputSink;
use crate::state::StateEstimator;

/// No-op sink. Used for sweep batches where only estimator state is needed.
pub struct NullSink;

impl OutputSink for NullSink {
    fn emit(&mut self, _: &Quantum, _: &dyn StateEstimator, _: &[EventId]) -> std::io::Result<()> {
        Ok(())
    }

    fn finish(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
