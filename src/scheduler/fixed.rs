//! Fixed scheduler: holds a constant set of counters for every quantum.

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;

/// Returns the same counter set every step; used by the `sweep` command to hold counters constant for an entire run.
pub struct FixedScheduler {
    /// The invariant set of events returned on every call to `next_step`.
    active: Vec<EventId>,
}

impl FixedScheduler {
    /// Creates a scheduler that always returns `active`.
    pub fn new(active: Vec<EventId>) -> Self {
        Self { active }
    }
}

impl Scheduler for FixedScheduler {
    fn init(
        &mut self,
        _all_events: Vec<EventId>,
        _num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        _estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        ScheduleDecision {
            active_events: self.active.clone(),
            duration: None,
        }
    }
}
