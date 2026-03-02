use crate::event_registry::{EventId, EventRegistry};
use crate::scheduler::{ScheduleDecision, Scheduler};
use std::time::Duration;

pub struct TestScheduler {
    events: Vec<EventId>,
    current_idx: usize,
}

impl TestScheduler {
    pub fn new(registry: &EventRegistry) -> Self {
        let target_names = vec![
            "all_data_cache_accesses",
            "ex_ret_instr",
            "ex_ret_brn_tkn",
            "ex_ret_brn",
            "fp_ret_sse_avx_ops.all",
        ];

        let mut events = Vec::new();
        for name in target_names {
            if let Some(id) = registry.lookup(name) {
                events.push(id);
            } else {
                eprintln!(
                    "[WARN] TestScheduler: Event '{}' not found in library",
                    name
                );
            }
        }

        Self {
            events,
            current_idx: 0,
        }
    }
}

impl Scheduler for TestScheduler {
    fn init(&mut self, _all_events: Vec<EventId>) {
        // No-op, we used the registry in new()
    }

    fn next_step(&mut self) -> ScheduleDecision {
        let chunk_size = 4;
        let len = self.events.len();

        if len == 0 {
            return ScheduleDecision {
                active_events: vec![],
                duration: None,
            };
        }

        let mut active = Vec::new();
        for i in 0..chunk_size {
            active.push(self.events[(self.current_idx + i) % len]);
        }

        self.current_idx = (self.current_idx + chunk_size) % len;

        ScheduleDecision {
            active_events: active,
            duration: Some(Duration::from_millis(10)), // 10ms for stability
        }
    }
}
