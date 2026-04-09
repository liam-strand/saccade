use crate::event_registry::EventId;
use crate::quantum::Quantum;
use crate::virtual_counter::VirtualCounterState;

pub mod csv;
pub mod null;
pub mod perfetto;

/// Pluggable output consumer. Each sink receives the full `Quantum` (raw samples +
/// lazy aggregates) and current VCS state once per profiler step.
pub trait OutputSink {
    fn emit(
        &mut self,
        quantum: &Quantum,
        vcs: &VirtualCounterState,
        active_set: &[EventId],
    ) -> std::io::Result<()>;

    fn finish(&mut self) -> std::io::Result<()>;
}
