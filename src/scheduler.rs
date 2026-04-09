pub mod distribution;
pub mod fixed;
pub mod random;
pub mod round_robin;
pub mod test;

use crate::event_registry::EventId;
use crate::quantum::Quantum;
use crate::virtual_counter::VirtualCounterState;
use std::time::Duration;

/// Pluggable counter selection policy.
pub trait Scheduler {
    /// Initialize with the universe of possible events and the number of hardware slots.
    fn init(&mut self, all_events: Vec<EventId>, num_slots: usize);

    /// Calculate the next set of events to monitor.
    ///
    /// Receives the full `Quantum` (raw samples + lazy aggregates) and the current
    /// VCS state (rate estimates + uncertainty). The active set returned must not
    /// exceed `num_slots` (as passed to `init`).
    fn next_step(&mut self, quantum: &Quantum, vcs: &VirtualCounterState) -> ScheduleDecision;
}

/// Output from the scheduler: what should we do next?
pub struct ScheduleDecision {
    /// The set of events to activate for the next window.
    pub active_events: Vec<EventId>,
    /// Optional override of the default step duration.
    pub duration: Option<Duration>,
}
