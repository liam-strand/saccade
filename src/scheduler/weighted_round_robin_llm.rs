use crate::event::EventId;
use crate::llm::LlmClient;
use crate::quantum::Quantum;
use crate::scheduler::llm_common;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;

pub struct WeightedRoundRobinLlmScheduler {
    client: LlmClient,
    event_info: Vec<(EventId, String, String)>,
    cycle: Vec<EventId>,
    step_idx: usize,
    num_slots: usize,
    guidance: Option<String>,
}

impl WeightedRoundRobinLlmScheduler {
    pub fn new(
        event_info: Vec<(EventId, String, String)>,
        client: LlmClient,
        guidance: Option<String>,
    ) -> Self {
        Self {
            client,
            event_info,
            cycle: Vec::new(),
            step_idx: 0,
            num_slots: 0,
            guidance,
        }
    }

    fn build_cycle(weights: &[(EventId, u32)]) -> Vec<EventId> {
        let gcd = weights.iter().map(|(_, w)| *w).fold(0u32, gcd_u32);
        let mut remaining: Vec<(EventId, u32)> = weights
            .iter()
            .map(|&(id, w)| (id, w / gcd.max(1)))
            .collect();
        // High-weight events lead within each round for more even distribution.
        remaining.sort_by(|a, b| b.1.cmp(&a.1));
        let total: u32 = remaining.iter().map(|(_, w)| *w).sum();
        let mut cycle = Vec::with_capacity(total as usize);
        loop {
            let mut added = false;
            for (id, rem) in &mut remaining {
                if *rem > 0 {
                    cycle.push(*id);
                    *rem -= 1;
                    added = true;
                }
            }
            if !added {
                break;
            }
        }
        cycle
    }
}

fn gcd_u32(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd_u32(b, a % b) }
}

impl Scheduler for WeightedRoundRobinLlmScheduler {
    fn init(
        &mut self,
        all_events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.num_slots = num_slots;
        let pb = llm_common::build_weights_prompt(
            &self.event_info,
            num_slots,
            self.guidance.as_deref(),
        );
        tracing::debug!(
            "WeightedRoundRobin system message:\n{}",
            pb.build()[0].content
        );
        let messages = pb.build().to_vec();
        let client = &self.client;
        let weights = llm_common::chat_with_retry(
            |m| client.chat(m),
            messages,
            |resp| llm_common::parse_weights_response(resp, &all_events),
            2,
        )?;
        let pairs: Vec<(EventId, u32)> =
            weights.iter().map(|ew| (ew.event_id, ew.weight)).collect();
        self.cycle = Self::build_cycle(&pairs);
        tracing::info!(
            "WeightedRoundRobinLlmScheduler: {} items in cycle",
            self.cycle.len()
        );
        Ok(())
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        _estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        let len = self.cycle.len();
        let mut active_events = Vec::with_capacity(self.num_slots);
        if len > 0 {
            for _ in 0..self.num_slots {
                active_events.push(self.cycle[self.step_idx]);
                self.step_idx = (self.step_idx + 1) % len;
            }
        }
        ScheduleDecision {
            active_events,
            duration: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::Quantum;
    use crate::state::{CounterEstimate, EstimateKey};
    use std::collections::HashMap;

    struct NullEstimator {
        estimates: HashMap<EstimateKey, CounterEstimate>,
    }

    impl NullEstimator {
        fn new() -> Self {
            Self {
                estimates: HashMap::new(),
            }
        }
    }

    impl crate::state::StateEstimator for NullEstimator {
        fn measurement_update(&mut self, _: u32, _: EventId, _: f64, _: f64, _: u32, _: u64) {}
        fn time_update(&mut self, _: u32, _: EventId, _: u64) {}
        fn rate(&self, _: u32, _: EventId) -> f64 {
            0.0
        }
        fn uncertainty(&self, _: u32, _: EventId) -> f64 {
            0.0
        }
        fn all_estimates(&self) -> &HashMap<EstimateKey, CounterEstimate> {
            &self.estimates
        }
    }

    fn empty_quantum() -> Quantum {
        Quantum::new(vec![], 0, 0)
    }

    fn make_scheduler(cycle: Vec<EventId>, num_slots: usize) -> WeightedRoundRobinLlmScheduler {
        let mut s = WeightedRoundRobinLlmScheduler::new(
            vec![],
            LlmClient::new("http://localhost:0", "test-model"),
            None,
        );
        s.cycle = cycle;
        s.num_slots = num_slots;
        s
    }

    #[test]
    fn build_cycle_proportional() {
        let weights = vec![(0, 3u32), (1, 1u32)];
        let cycle = WeightedRoundRobinLlmScheduler::build_cycle(&weights);
        assert_eq!(cycle.len(), 4);
        assert_eq!(cycle.iter().filter(|&&id| id == 0).count(), 3);
        assert_eq!(cycle.iter().filter(|&&id| id == 1).count(), 1);
    }

    #[test]
    fn build_cycle_single_event() {
        let weights = vec![(0, 5u32)];
        let cycle = WeightedRoundRobinLlmScheduler::build_cycle(&weights);
        assert_eq!(cycle.len(), 1);
        assert_eq!(cycle[0], 0);
    }

    #[test]
    fn build_cycle_gcd_normalization() {
        let weights = vec![(0, 4u32), (1, 4u32), (2, 4u32)];
        let cycle = WeightedRoundRobinLlmScheduler::build_cycle(&weights);
        assert_eq!(cycle.len(), 3);
    }

    #[test]
    fn next_step_cycles() {
        let mut s = make_scheduler(vec![0, 1, 2], 2);
        let q = empty_quantum();
        let est = NullEstimator::new();

        let d1 = s.next_step(&q, &est);
        assert_eq!(d1.active_events, vec![0, 1]);
        assert!(d1.duration.is_none());

        let d2 = s.next_step(&q, &est);
        assert_eq!(d2.active_events, vec![2, 0]);

        let d3 = s.next_step(&q, &est);
        assert_eq!(d3.active_events, vec![1, 2]);
    }

    #[test]
    fn next_step_wraps_on_short_cycle() {
        let mut s = make_scheduler(vec![0, 1], 4);
        let q = empty_quantum();
        let est = NullEstimator::new();

        let d = s.next_step(&q, &est);
        assert_eq!(d.active_events, vec![0, 1, 0, 1]);
    }

    #[test]
    fn next_step_empty_cycle_returns_empty() {
        let mut s = make_scheduler(vec![], 2);
        let q = empty_quantum();
        let est = NullEstimator::new();

        let d = s.next_step(&q, &est);
        assert!(d.active_events.is_empty());
    }

    #[test]
    #[ignore = "requires network access to dubliner.cs.northwestern.edu"]
    fn llm_generates_parseable_weights() {
        let event_info = vec![
            (0u32, "cache-misses".to_string(), "Cache misses".to_string()),
            (1u32, "branch-misses".to_string(), "Branch mispredictions".to_string()),
            (2u32, "instructions".to_string(), "Instructions retired".to_string()),
        ];
        let all_events: Vec<u32> = event_info.iter().map(|(id, _, _)| *id).collect();
        let client = LlmClient::new("http://dubliner.cs.northwestern.edu:11434", "gemma4");

        let pb = llm_common::build_weights_prompt(&event_info, 2, None);
        let response = client.chat(pb.build()).expect("LLM call should succeed");
        eprintln!("LLM weights response:\n{response}");

        let weights = llm_common::parse_weights_response(&response, &all_events)
            .expect("should parse into valid weights");
        assert_eq!(weights.len(), all_events.len());
        for ew in &weights {
            assert!(ew.weight >= 1 && ew.weight <= 10);
        }
    }
}
