use rand::prelude::*;

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::virtual_counter::VirtualCounterState;

pub struct RandomScheduler {
    events: Vec<EventId>,
    num_slots: usize,
    rng: ThreadRng,
}

impl RandomScheduler {
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
    fn init(&mut self, all_events: Vec<EventId>, num_slots: usize) {
        self.events = all_events;
        self.num_slots = num_slots;
    }

    fn next_step(&mut self, _quantum: &Quantum, _vcs: &VirtualCounterState) -> ScheduleDecision {
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
