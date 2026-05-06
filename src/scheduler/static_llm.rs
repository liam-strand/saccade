use serde::{Deserialize, Serialize};

use crate::event::EventId;
use crate::llm::{LlmClient, PromptBuilder};
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;
use std::collections::HashSet;
use std::time::Duration;

#[derive(Debug, Deserialize, Serialize)]
struct ScheduleStep {
    duration_ms: u64,
    events: Vec<EventId>,
}

pub struct StaticLlmScheduler {
    client: LlmClient,
    event_info: Vec<(EventId, String, String)>,
    schedule: Vec<ScheduleStep>,
    step_idx: usize,
    all_events: Vec<EventId>,
    num_slots: usize,
}

impl StaticLlmScheduler {
    pub fn new(event_info: Vec<(EventId, String, String)>, client: LlmClient) -> Self {
        Self {
            client,
            event_info,
            schedule: Vec::new(),
            step_idx: 0,
            all_events: Vec::new(),
            num_slots: 0,
        }
    }

    fn build_prompt(&self) -> PromptBuilder {
        let mut event_list = String::new();
        for (id, name, desc) in &self.event_info {
            event_list.push_str(&format!("  {id}: {name} — {desc}\n"));
        }
        let num_slots = self.num_slots;
        // Concrete example avoids placeholder ambiguity for small local models.
        let example = serde_json::to_string_pretty(&vec![
            ScheduleStep {
                duration_ms: 25,
                events: vec![0, 2, 4, 6],
            },
            ScheduleStep {
                duration_ms: 20,
                events: vec![1, 3, 24, 35],
            },
            ScheduleStep {
                duration_ms: 30,
                events: vec![5, 54, 90, 66],
            },
        ])
        .expect("should serialize");

        let system = format!(
            "You are an expert Linux performance profiling assistant. \
             You generate hardware performance counter observation schedules \
             for a sampling profiler that activates {num_slots} \
             counters simultaneously. The profiler cycles through the schedule \
             indefinitely, rotating which counters are active to build a \
             complete picture of program behavior over time."
        );
        let user = format!(
            "Available performance counters (format — ID: name: description):
            {event_list}
            Generate a cyclic measurement schedule that covers all of the \
            counters above. Each step activates {num_slots} counters \
            and runs for a specified duration (aim for 10–1000 ms per step). \
            Prioritize counters that reveal common bottlenecks; cache misses, \
            TLB pressure, branch mispredictions, memory stalls; and ensure \
            every counter appears at least once across the full cycle.

            Example output (hypothetical IDs):
            {example}

            Produce the schedule for the counters listed above. \
            Your entire response must be a single JSON array. \
            No explanation, no prose, no markdown fences."
        );
        PromptBuilder::new().system(system).user(user)
    }

    fn parse_schedule(&self, response: &str) -> Result<Vec<ScheduleStep>, String> {
        let start = response
            .find('[')
            .ok_or_else(|| format!("no JSON array found in LLM response:\n{response}"))?;
        let end = response
            .rfind(']')
            .ok_or_else(|| format!("JSON array is not closed in LLM response:\n{response}"))?;
        if end < start {
            return Err(format!("malformed JSON array in LLM response:\n{response}"));
        }

        let mut steps: Vec<ScheduleStep> =
            serde_json::from_str(&response[start..=end]).map_err(|e| {
                format!("failed to parse LLM schedule JSON: {e}\nresponse:\n{response}")
            })?;

        if steps.is_empty() {
            return Err("LLM returned an empty schedule".to_string());
        }

        let valid_ids: HashSet<u32> = self.all_events.iter().copied().collect();
        for (i, step) in steps.iter_mut().enumerate() {
            step.events.retain(|id| valid_ids.contains(id));
            step.events.truncate(self.num_slots);
            step.duration_ms = step.duration_ms.max(1);
            if step.events.is_empty() {
                return Err(format!("step {i} contains no valid event IDs"));
            }
        }

        Ok(steps)
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

    fn make_scheduler(all_events: Vec<EventId>, num_slots: usize) -> StaticLlmScheduler {
        let mut s =
            StaticLlmScheduler::new(vec![], LlmClient::new("http://localhost:0", "test-model"));
        s.all_events = all_events;
        s.num_slots = num_slots;
        s
    }

    fn empty_quantum() -> Quantum {
        Quantum::new(vec![], 0, 0)
    }

    #[test]
    fn parse_valid_schedule() {
        let s = make_scheduler(vec![0, 1, 2, 3, 4], 4);
        let json = r#"[
            {"duration_ms": 25, "events": [0, 1]},
            {"duration_ms": 30, "events": [2, 3, 4]}
        ]"#;
        let steps = s.parse_schedule(json).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].duration_ms, 25);
        assert_eq!(steps[0].events, vec![0, 1]);
        assert_eq!(steps[1].events, vec![2, 3, 4]);
    }

    #[test]
    fn parse_no_array() {
        let s = make_scheduler(vec![0, 1], 4);
        let err = s.parse_schedule("the model just said hello").unwrap_err();
        assert!(err.contains("no JSON array found"));
    }

    #[test]
    fn parse_empty_array() {
        let s = make_scheduler(vec![0, 1], 4);
        let err = s.parse_schedule("[]").unwrap_err();
        assert!(err.contains("empty"));
    }

    #[test]
    fn parse_filters_invalid_ids() {
        let s = make_scheduler(vec![0, 1], 4);
        let json = r#"[{"duration_ms": 20, "events": [0, 99]}]"#;
        let steps = s.parse_schedule(json).unwrap();
        assert_eq!(steps[0].events, vec![0]);
    }

    #[test]
    fn parse_all_invalid_ids_errors() {
        let s = make_scheduler(vec![0, 1], 4);
        let json = r#"[{"duration_ms": 20, "events": [99, 100]}]"#;
        let err = s.parse_schedule(json).unwrap_err();
        assert!(err.contains("no valid event IDs"));
    }

    #[test]
    fn parse_clamps_zero_duration() {
        let s = make_scheduler(vec![0], 4);
        let json = r#"[{"duration_ms": 0, "events": [0]}]"#;
        let steps = s.parse_schedule(json).unwrap();
        assert_eq!(steps[0].duration_ms, 1);
    }

    #[test]
    fn parse_truncates_to_num_slots() {
        let s = make_scheduler(vec![0, 1, 2, 3, 4], 2);
        let json = r#"[{"duration_ms": 10, "events": [0, 1, 2, 3, 4]}]"#;
        let steps = s.parse_schedule(json).unwrap();
        assert_eq!(steps[0].events.len(), 2);
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

        let client = LlmClient::new("http://dubliner.cs.northwestern.edu:11434", "gemma4");
        let mut s = StaticLlmScheduler::new(event_info, client);
        s.all_events = all_events;
        s.num_slots = 4;

        let pb = s.build_prompt();
        let response = s.client.chat(pb.build()).expect("LLM call should succeed");
        eprintln!("LLM response:\n{response}");

        let steps = s
            .parse_schedule(&response)
            .expect("should parse into a valid schedule");

        assert!(!steps.is_empty());
        for step in &steps {
            assert!(!step.events.is_empty());
            assert!(step.duration_ms >= 1);
        }
    }

    #[test]
    fn next_step_cycles() {
        let mut s = make_scheduler(vec![0, 1, 2], 4);
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
}

impl Scheduler for StaticLlmScheduler {
    fn init(&mut self, all_events: Vec<EventId>, num_slots: usize) {
        self.all_events = all_events;
        self.num_slots = num_slots;

        let response = {
            let pb = self.build_prompt();
            self.client.chat(pb.build())
        }
        .unwrap_or_else(|e| panic!("StaticLlmScheduler: LLM call failed: {e}"));

        self.schedule = self
            .parse_schedule(&response)
            .unwrap_or_else(|e| panic!("StaticLlmScheduler: bad schedule from LLM: {e}"));

        tracing::info!("StaticLlmScheduler: {} steps in cycle", self.schedule.len());
    }

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
