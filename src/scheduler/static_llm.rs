//! Scheduler that generates one fixed cyclic counter schedule via an LLM at startup and then
//! replays it indefinitely without further model calls.

use crate::event::EventId;
use crate::llm::{LlmClient, LlmLatencyProfile};
use crate::quantum::Quantum;
use crate::scheduler::llm_common::{self, ScheduleStep};
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;
use std::time::Duration;

/// LLM-based scheduler whose counter rotation schedule is generated once at `init` and never changed.
pub struct StaticLlmScheduler {
    /// HTTP client used to call the LLM.
    client: LlmClient,
    /// Metadata for every available hardware event: `(id, name, description)`.
    event_info: Vec<(EventId, String, String)>,
    /// The cyclic schedule returned by the LLM.
    schedule: Vec<ScheduleStep>,
    /// Index of the next step to serve from `schedule`.
    step_idx: usize,
    /// All valid event IDs as reported by `init`.
    all_events: Vec<EventId>,
    /// Number of hardware counter slots the profiler can activate simultaneously.
    num_slots: usize,
    /// Optional natural-language guidance forwarded to the LLM system message.
    guidance: Option<String>,
    /// Optional latency profile for overriding measured LLM call latency in simulation.
    latency_profile: Option<LlmLatencyProfile>,
}

impl StaticLlmScheduler {
    /// Create a new scheduler; `init` must be called before `next_step` to populate the schedule.
    pub fn new(
        event_info: Vec<(EventId, String, String)>,
        client: LlmClient,
        guidance: Option<String>,
        latency_profile: Option<LlmLatencyProfile>,
    ) -> Self {
        Self {
            client,
            event_info,
            schedule: Vec::new(),
            step_idx: 0,
            all_events: Vec::new(),
            num_slots: 0,
            guidance,
            latency_profile,
        }
    }
}

impl Scheduler for StaticLlmScheduler {
    /// Call the LLM to generate the cyclic schedule; must complete before `next_step` is called.
    fn init(
        &mut self,
        all_events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.all_events = all_events;
        self.num_slots = num_slots;

        let sampled = self.latency_profile.as_mut().and_then(|p| p.sample("static_setup"));
        self.schedule = {
            let pb = llm_common::build_init_prompt(
                &self.event_info,
                num_slots,
                self.guidance.as_deref(),
            );
            tracing::debug!("StaticLlm system message:\n{}", pb.build()[0].content);
            let messages = pb.build().to_vec();
            let client = &self.client;
            let all_events = &self.all_events;
            let schema = llm_common::schedule_json_schema(num_slots, all_events);
            llm_common::chat_with_retry(
                |m| client.chat(m, "schedule", &schema, "static_setup", sampled),
                messages,
                |resp| llm_common::parse_schedule_response(resp, all_events, num_slots),
                2,
            )
        }?;

        tracing::info!("StaticLlmScheduler: {} steps in cycle", self.schedule.len());
        Ok(())
    }

    /// Return the next step from the fixed schedule, wrapping around to the beginning when exhausted.
    fn next_step(
        &mut self,
        _quantum: &Quantum,
        _estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        let step = &self.schedule[self.step_idx];
        let result = ScheduleDecision {
            active_events: step.events.clone(),
            duration: Some(Duration::from_millis(step.duration_ms)),
        };
        self.step_idx = (self.step_idx + 1) % self.schedule.len();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::Quantum;
    use crate::scheduler::llm_common::ScheduleStep;
    use crate::state::{CounterEstimate, EstimateKey};
    use std::collections::HashMap;

    /// A `StateEstimator` stub that always reports zero rate and uncertainty.
    struct NullEstimator {
        /// Empty estimate map; never populated by the stub.
        estimates: HashMap<EstimateKey, CounterEstimate>,
    }

    impl NullEstimator {
        /// Create an estimator with no observations.
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

    /// Produce a `Quantum` with no samples, suitable as a placeholder argument in scheduler tests.
    fn empty_quantum() -> Quantum {
        Quantum::new(vec![], 0, 0)
    }

    #[test]
    fn next_step_cycles() {
        let mut s = StaticLlmScheduler::new(
            vec![],
            LlmClient::new("http://localhost:0", "test-model", None),
            None,
            None,
        );
        s.all_events = vec![0, 1, 2];
        s.num_slots = 4;
        s.schedule = vec![
            ScheduleStep {
                duration_ms: 10,
                events: vec![0],
            },
            ScheduleStep {
                duration_ms: 20,
                events: vec![1],
            },
            ScheduleStep {
                duration_ms: 30,
                events: vec![2],
            },
        ];

        let q = empty_quantum();
        let est = NullEstimator::new();

        let d1 = s.next_step(&q, &est);
        assert_eq!(d1.active_events, vec![0]);
        assert_eq!(d1.duration, Some(Duration::from_millis(10)));

        let d2 = s.next_step(&q, &est);
        assert_eq!(d2.active_events, vec![1]);

        let d3 = s.next_step(&q, &est);
        assert_eq!(d3.active_events, vec![2]);

        let d4 = s.next_step(&q, &est);
        assert_eq!(d4.active_events, vec![0]);
    }

    #[test]
    #[ignore = "requires network access to dubliner.cs.northwestern.edu"]
    fn llm_generates_parseable_schedule() {
        let event_info = vec![
            (0u32, "cache-misses".to_string(), "Cache misses".to_string()),
            (
                1u32,
                "cache-references".to_string(),
                "Cache accesses, including hits".to_string(),
            ),
            (
                2u32,
                "branch-misses".to_string(),
                "Mispredicted branches".to_string(),
            ),
            (
                3u32,
                "branch-instructions".to_string(),
                "Branch instructions retired".to_string(),
            ),
            (
                4u32,
                "instructions".to_string(),
                "Instructions retired".to_string(),
            ),
            (
                5u32,
                "cpu-cycles".to_string(),
                "Total CPU cycles".to_string(),
            ),
            (
                6u32,
                "dTLB-load-misses".to_string(),
                "Data TLB load misses".to_string(),
            ),
            (
                7u32,
                "dTLB-store-misses".to_string(),
                "Data TLB store misses".to_string(),
            ),
        ];
        let all_events: Vec<u32> = event_info.iter().map(|(id, _, _)| *id).collect();

        let client = LlmClient::new("http://dubliner.cs.northwestern.edu:11434", "gemma4", None);
        let mut s = StaticLlmScheduler::new(event_info, client, None, None);
        s.all_events = all_events.clone();
        s.num_slots = 4;

        let pb = llm_common::build_init_prompt(&s.event_info, s.num_slots, None);
        let schema = llm_common::schedule_json_schema(s.num_slots, &s.all_events);
        let response = s.client.chat(pb.build(), "schedule", &schema, "static_setup", None).expect("LLM call should succeed");
        eprintln!("LLM response:\n{response}");

        let steps = llm_common::parse_schedule_response(&response, &all_events, s.num_slots)
            .expect("should parse into a valid schedule");

        assert!(!steps.is_empty());
        for step in &steps {
            assert!(!step.events.is_empty());
            assert!(step.duration_ms >= 1);
        }
    }
}
