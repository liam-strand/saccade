use crate::event::{EventId, EventRegistry};
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::virtual_counter::VirtualCounterState;
use std::time::Duration;

pub struct TestScheduler {
    events: Vec<EventId>,
    num_slots: usize,
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
            num_slots: 4,
            current_idx: 0,
        }
    }
}

impl Scheduler for TestScheduler {
    fn init(&mut self, _all_events: Vec<EventId>, num_slots: usize) {
        self.num_slots = num_slots;
    }

    fn next_step(&mut self, _quantum: &Quantum, _vcs: &VirtualCounterState) -> ScheduleDecision {
        let len = self.events.len();

        if len == 0 {
            return ScheduleDecision {
                active_events: vec![],
                duration: None,
            };
        }

        let mut active = Vec::new();
        for i in 0..self.num_slots {
            active.push(self.events[(self.current_idx + i) % len]);
        }

        self.current_idx = (self.current_idx + self.num_slots) % len;

        ScheduleDecision {
            active_events: active,
            duration: Some(Duration::from_millis(10)),
        }
    }
}
