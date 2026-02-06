use rand::prelude::*;

use crate::event_registry::EventId;
use crate::scheduler::{ScheduleDecision, Scheduler};

pub struct RandomScheduler {
    events: Vec<EventId>,
    rng: ThreadRng,
}

impl RandomScheduler {
    fn new() -> Self {
        return Self {
            events: Vec::new(),
            rng: rand::rng(),
        };
    }
}

impl Default for RandomScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler for RandomScheduler {
    fn init(&mut self, all_events: &[EventId]) {
        self.events = all_events.to_vec();
    }
    fn next_step(&mut self) -> ScheduleDecision {
        ScheduleDecision {
            active_events: self
                .events
                .choose_multiple(&mut self.rng, 4)
                .cloned()
                .collect(),
            duration: None,
        }
    }
}
