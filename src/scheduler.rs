pub mod distribution;
pub mod random;
pub mod round_robin;
pub mod test;

use crate::event_registry::EventId;
use crate::virtual_counter::VirtualCounterState;
use std::time::Duration;

/// The main Policy interface
pub trait Scheduler {
    /// Initialize the scheduler with the universe of possible events.
    /// This is called once at startup.
    fn init(&mut self, all_events: Vec<EventId>);

    /// Calculate the next set of events to monitor.
    /// `state` provides rate estimates and uncertainty for all counters.
    fn next_step(&mut self, state: &VirtualCounterState) -> ScheduleDecision;
}

/// Output from the scheduler: what should we do next?
pub struct ScheduleDecision {
    /// The set of events to activate for the next window
    pub active_events: Vec<EventId>,
    /// How long to keep this window active before querying again
    /// Use None for "default" or "until interrupt"
    pub duration: Option<Duration>,
}
