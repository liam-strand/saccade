use crate::event_registry::EventId;
use crate::scheduler::ScheduleDecision;
use crate::scheduler::Scheduler;
use crate::virtual_counter::VirtualCounterState;

pub struct RoundRobinScheduler {
    events: Vec<EventId>,
    current: usize,
}

impl Default for RoundRobinScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRobinScheduler {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            current: 0,
        }
    }
}

impl Scheduler for RoundRobinScheduler {
    fn init(&mut self, events: Vec<EventId>) {
        self.events = events;
    }

    fn next_step(&mut self, _state: &VirtualCounterState) -> ScheduleDecision {
        let mut active_events = Vec::with_capacity(4);
        let len = self.events.len();

        if len > 0 {
            for _ in 0..4 {
                active_events.push(self.events[self.current]);
                self.current = (self.current + 1) % len;
            }
        }

        ScheduleDecision {
            active_events,
            duration: None,
        }
    }
}
