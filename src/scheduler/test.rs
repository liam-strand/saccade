//! Hardcoded test scheduler cycling through a fixed set of AMD performance events.

use crate::event::{EventId, EventRegistry};
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;
use std::time::Duration;

/// Round-robin scheduler over a hardcoded subset of AMD events, used for manual integration testing.
pub struct TestScheduler {
    /// Resolved IDs of the target events looked up from the registry at construction time.
    events: Vec<EventId>,
    /// Number of counters to activate per step, overridden by `init`.
    num_slots: usize,
    /// Index of the first event in `events` to activate in the next step.
    current_idx: usize,
}

impl TestScheduler {
    /// Resolves a fixed list of AMD event names from `registry` and builds the scheduler.
    ///
    /// Events not found in the registry are skipped with a warning printed to stderr.
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
            num_slots: 6,
            current_idx: 0,
        }
    }
}

impl Scheduler for TestScheduler {
    fn init(
        &mut self,
        _all_events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.num_slots = num_slots;
        Ok(())
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        _estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
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
