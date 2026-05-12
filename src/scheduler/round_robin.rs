use crate::event::EventId;
use crate::quantum::Quantum;
use crate::scheduler::ScheduleDecision;
use crate::scheduler::Scheduler;
use crate::state::StateEstimator;

pub struct RoundRobinScheduler {
    events: Vec<EventId>,
    num_slots: usize,
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
            num_slots: 4,
            current: 0,
        }
    }
}

impl Scheduler for RoundRobinScheduler {
    fn init(
        &mut self,
        events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.events = events;
        self.num_slots = num_slots;
        Ok(())
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        _estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        let mut active_events = Vec::with_capacity(self.num_slots);
        let len = self.events.len();

        if len > 0 {
            for _ in 0..self.num_slots {
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
