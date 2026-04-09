use crate::event::EventId;
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::virtual_counter::VirtualCounterState;

/// Always returns the same fixed set of counters.
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
    fn init(&mut self, _all_events: Vec<EventId>, _num_slots: usize) {}

    fn next_step(&mut self, _quantum: &Quantum, _vcs: &VirtualCounterState) -> ScheduleDecision {
        ScheduleDecision {
            active_events: self.active.clone(),
            duration: None,
        }
    }
}
