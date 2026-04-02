use crate::event_registry::EventId;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::virtual_counter::VirtualCounterState;

/// A scheduler that always returns the same fixed set of counters.
/// Used by the `sweep` command to hold counters constant for an entire run.
pub struct FixedScheduler {
    active: Vec<EventId>,
}

impl FixedScheduler {
    pub fn new(active: Vec<EventId>) -> Self {
        Self { active }
    }
}

impl Scheduler for FixedScheduler {
    fn init(&mut self, _all_events: Vec<EventId>) {}

    fn next_step(&mut self, _state: &VirtualCounterState) -> ScheduleDecision {
        ScheduleDecision {
            active_events: self.active.clone(),
            duration: None,
        }
    }
}
