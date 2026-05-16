//! No-op output sink that discards all profiling data.

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::sink::OutputSink;
use crate::state::StateEstimator;

/// Discards every quantum without producing any output; used when only estimator state is needed.
pub struct NullSink;

impl OutputSink for NullSink {
    fn emit(&mut self, _: &Quantum, _: &dyn StateEstimator, _: &[EventId]) -> std::io::Result<()> {
        Ok(())
    }

    fn finish(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
