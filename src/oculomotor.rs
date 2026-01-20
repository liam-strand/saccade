use crate::event_registry::EventId;
use crate::event_registry::EventRegistry;
use crate::scheduler::ScheduleDecision;
use crate::scheduler::Scheduler;

pub struct Oculomotor<S: Scheduler> {
    registry: EventRegistry,
    scheduler: S,
    active_set: Vec<EventId>,
}

impl<S: Scheduler> Oculomotor<S> {
    pub fn new(registry: EventRegistry, scheduler: S) -> Self {
        Self {
            registry,
            scheduler,
            active_set: Vec::new(),
        }
    }

    pub fn step(&mut self) -> Option<std::time::Duration> {
        let decision = self.scheduler.next_step();
        self.apply_schedule(decision)
    }

    fn apply_schedule(&mut self, decision: ScheduleDecision) -> Option<std::time::Duration> {
        for event_id in decision.active_events.iter() {
            if self.active_set.contains(event_id) {
                continue;
            }
            self.registry.enable(*event_id);
            self.active_set.push(*event_id);
        }
        for event_id in self
            .active_set
            .iter()
            .filter(|id| !decision.active_events.contains(id))
        {
            self.registry.disable(*event_id);
        }
        self.active_set
            .retain(|id| decision.active_events.contains(id));

        decision.duration
    }
}
