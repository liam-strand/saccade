use crate::event::{Event, EventId};
use crate::llm::LlmClient;
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;

#[allow(dead_code)]
pub struct StaticLlmScheduler {
    client: LlmClient,
    event_info: Vec<Event>,
    all_events: Vec<EventId>,
    num_slots: usize,
}

impl StaticLlmScheduler {
    pub fn new(event_info: Vec<Event>, client: LlmClient) -> Self {
        Self {
            client,
            event_info,
            all_events: Vec::new(),
            num_slots: 0,
        }
    }
}

impl Scheduler for StaticLlmScheduler {
    fn init(&mut self, all_events: Vec<EventId>, num_slots: usize) {
        self.all_events = all_events;
        self.num_slots = num_slots;
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        _estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        let active_events = self
            .all_events
            .iter()
            .take(self.num_slots)
            .cloned()
            .collect();
        ScheduleDecision {
            active_events,
            duration: None,
        }
    }
}
