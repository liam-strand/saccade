//! Random scheduler: picks a uniformly random subset of counters each step.

use rand::prelude::*;

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;

/// Samples `num_slots` events uniformly at random from the full event list each step.
pub struct RandomScheduler {
    /// Full set of candidate events, set by `init`.
    events: Vec<EventId>,
    /// Number of counters to sample per step, overridden by `init`.
    num_slots: usize,
    /// Thread-local RNG used for sampling.
    rng: ThreadRng,
}

impl RandomScheduler {
    /// Creates a scheduler with an empty event list and a default slot count of 4.
    fn new() -> Self {
        Self {
            events: Vec::new(),
            num_slots: 4,
            rng: rand::rng(),
        }
    }
}

impl Default for RandomScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler for RandomScheduler {
    fn init(
        &mut self,
        all_events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.events = all_events;
        self.num_slots = num_slots;
        Ok(())
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        _estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        ScheduleDecision {
            active_events: self
                .events
                .choose_multiple(&mut self.rng, self.num_slots)
                .cloned()
                .collect(),
            duration: None,
        }
    }
}
