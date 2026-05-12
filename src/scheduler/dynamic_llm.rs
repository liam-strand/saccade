use crate::event::EventId;
use crate::llm::{ChatMessage, LlmClient, PromptBuilder};
use crate::quantum::Quantum;
use crate::scheduler::llm_common::{self, ScheduleStep};
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

pub struct DynamicLlmScheduler {
    client: LlmClient,
    event_info: Vec<(EventId, String, String)>,
    schedule: Vec<ScheduleStep>,
    step_idx: usize,
    all_events: Vec<EventId>,
    num_slots: usize,
    update_interval: u32,
    steps_since_update: u32,
    request_tx: Option<mpsc::SyncSender<Vec<ChatMessage>>>,
    result_rx: Option<mpsc::Receiver<Result<Vec<ScheduleStep>, String>>>,
}

impl DynamicLlmScheduler {
    pub fn new(
        event_info: Vec<(EventId, String, String)>,
        client: LlmClient,
        update_interval: u32,
    ) -> Self {
        Self {
            client,
            event_info,
            schedule: Vec::new(),
            step_idx: 0,
            all_events: Vec::new(),
            num_slots: 0,
            update_interval,
            steps_since_update: 0,
            request_tx: None,
            result_rx: None,
        }
    }

    fn build_update_prompt(&self, estimator: &dyn StateEstimator) -> PromptBuilder {
        let mut event_list = String::new();
        for (id, name, desc) in &self.event_info {
            event_list.push_str(&format!("  {id}: {name} — {desc}\n"));
        }

        // Aggregate estimates across threads: sum rates, max uncertainty, sum samples.
        let mut agg_rate: HashMap<EventId, f64> = HashMap::new();
        let mut agg_uncertainty: HashMap<EventId, f64> = HashMap::new();
        let mut agg_samples: HashMap<EventId, u64> = HashMap::new();
        for (&(_tid, event_id), est) in estimator.all_estimates() {
            *agg_rate.entry(event_id).or_default() += est.rate;
            let u = agg_uncertainty.entry(event_id).or_default();
            *u = u.max(est.uncertainty);
            *agg_samples.entry(event_id).or_default() += est.sample_count;
        }

        let id_to_name: HashMap<EventId, &str> = self
            .event_info
            .iter()
            .map(|(id, name, _)| (*id, name.as_str()))
            .collect();

        let mut obs_list = String::new();
        for &id in &self.all_events {
            let name = id_to_name.get(&id).copied().unwrap_or("unknown");
            match agg_rate.get(&id) {
                Some(&rate) => {
                    let uncertainty = agg_uncertainty[&id];
                    let samples = agg_samples[&id];
                    obs_list.push_str(&format!(
                        "  {id} ({name}): rate={rate:.3e} ev/ns, \
                         uncertainty={uncertainty:.3}, samples={samples}\n"
                    ));
                }
                None => obs_list.push_str(&format!("  {id} ({name}): not yet observed\n")),
            }
        }

        let current_schedule =
            serde_json::to_string_pretty(&self.schedule).expect("should serialize");
        let num_slots = self.num_slots;

        let system = format!(
            "You are an expert Linux performance profiling assistant. \
             You generate hardware performance counter observation schedules \
             for a sampling profiler that activates {num_slots} \
             counters simultaneously. The profiler cycles through the schedule \
             indefinitely, rotating which counters are active."
        );
        let user = format!(
            "Available performance counters (format — ID: name: description):
{event_list}
Recent observations (rate in events/ns, uncertainty in [0=confident, 1=unknown]):
{obs_list}
Current schedule (for reference):
{current_schedule}

Generate an updated cyclic schedule for the same {num_slots}-slot profiler. \
Prioritize events with high uncertainty or high rate — they are most informative. \
Cover all counters at least once across the full cycle.

Your entire response must be a single JSON array. \
No explanation, no prose, no markdown fences."
        );
        PromptBuilder::new().system(system).user(user)
    }
}

impl Scheduler for DynamicLlmScheduler {
    fn init(
        &mut self,
        all_events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.all_events = all_events;
        self.num_slots = num_slots;

        let response = {
            let pb = llm_common::build_init_prompt(&self.event_info, num_slots);
            self.client.chat(pb.build())
        }?;

        self.schedule =
            llm_common::parse_schedule_response(&response, &self.all_events, self.num_slots)
                .map_err(std::io::Error::other)?;

        tracing::info!(
            "DynamicLlmScheduler: {} steps in initial cycle",
            self.schedule.len()
        );

        // Spawn a persistent worker thread for background schedule updates.
        let (req_tx, req_rx) = mpsc::sync_channel::<Vec<ChatMessage>>(1);
        let (res_tx, res_rx) = mpsc::sync_channel(1);

        let client = self.client.clone();
        let all_events = self.all_events.clone();
        let num_slots = self.num_slots;
        std::thread::spawn(move || {
            for messages in req_rx {
                let result = client
                    .chat(&messages)
                    .map_err(|e| e.to_string())
                    .and_then(|resp| {
                        llm_common::parse_schedule_response(&resp, &all_events, num_slots)
                    });
                if res_tx.send(result).is_err() {
                    break;
                }
            }
        });

        self.request_tx = Some(req_tx);
        self.result_rx = Some(res_rx);
        Ok(())
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        // 1. Check for a completed background update.
        if let Some(rx) = &self.result_rx {
            match rx.try_recv() {
                Ok(Ok(steps)) => {
                    tracing::info!("DynamicLlmScheduler: updated to {} steps", steps.len());
                    self.schedule = steps;
                    self.step_idx = 0;
                }
                Ok(Err(msg)) => {
                    tracing::warn!("DynamicLlmScheduler: update failed, keeping schedule: {msg}");
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    tracing::warn!("DynamicLlmScheduler: worker thread exited");
                }
            }
        }

        // 2. Send a new update request if the interval has elapsed.
        // Reset the counter only on successful send; if the worker is busy,
        // leave the counter alone so we retry on the next step.
        self.steps_since_update += 1;
        if self.steps_since_update >= self.update_interval {
            let messages = self.build_update_prompt(estimator).build().to_vec();
            if let Some(tx) = &self.request_tx
                && tx.try_send(messages).is_ok()
            {
                self.steps_since_update = 0;
            }
        }

        // 3. Serve the current schedule step.
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

    struct MockEstimator {
        estimates: HashMap<EstimateKey, CounterEstimate>,
    }

    impl MockEstimator {
        fn new() -> Self {
            Self {
                estimates: HashMap::new(),
            }
        }

        fn add(&mut self, tid: u32, event_id: EventId, rate: f64, uncertainty: f64) {
            self.estimates.insert(
                (tid, event_id),
                CounterEstimate {
                    rate,
                    uncertainty,
                    ..Default::default()
                },
            );
        }
    }

    impl crate::state::StateEstimator for MockEstimator {
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

    fn make_scheduler(update_interval: u32) -> DynamicLlmScheduler {
        let mut s = DynamicLlmScheduler::new(
            vec![
                (0, "cache-misses".into(), "Cache misses".into()),
                (1, "branch-misses".into(), "Branch mispredictions".into()),
                (2, "instructions".into(), "Instructions retired".into()),
            ],
            LlmClient::new("http://localhost:0", "test-model"),
            update_interval,
        );
        s.all_events = vec![0, 1, 2];
        s.num_slots = 2;
        s.schedule = vec![
            ScheduleStep {
                duration_ms: 10,
                events: vec![0, 1],
            },
            ScheduleStep {
                duration_ms: 20,
                events: vec![2, 0],
            },
        ];
        s
    }

    #[test]
    fn next_step_cycles_without_update() {
        let mut s = make_scheduler(100);
        let q = empty_quantum();
        let est = MockEstimator::new();

        let d1 = s.next_step(&q, &est);
        assert_eq!(d1.active_events, vec![0, 1]);

        let d2 = s.next_step(&q, &est);
        assert_eq!(d2.active_events, vec![2, 0]);

        let d3 = s.next_step(&q, &est);
        assert_eq!(d3.active_events, vec![0, 1]);
    }

    #[test]
    fn build_update_prompt_includes_observations() {
        let s = make_scheduler(100);
        let mut est = MockEstimator::new();
        est.add(0, 0, 1.5e-6, 0.2);
        est.add(0, 1, 3.0e-7, 0.8);

        let pb = s.build_update_prompt(&est);
        let messages = pb.build();
        let user_content = &messages.iter().find(|m| m.role == "user").unwrap().content;

        assert!(
            user_content.contains("not yet observed"),
            "event 2 has no estimate"
        );
        assert!(user_content.contains("1.500e-6") || user_content.contains("1.5e-6"));
        assert!(user_content.contains("Current schedule"));
    }

    #[test]
    #[ignore = "requires network access to dubliner.cs.northwestern.edu"]
    fn llm_generates_parseable_update() {
        let mut s = DynamicLlmScheduler::new(
            vec![
                (0, "cache-misses".into(), "Cache misses".into()),
                (1, "branch-misses".into(), "Branch mispredictions".into()),
                (2, "instructions".into(), "Instructions retired".into()),
            ],
            LlmClient::new("http://dubliner.cs.northwestern.edu:11434", "gemma4"),
            1,
        );
        s.all_events = vec![0, 1, 2];
        s.num_slots = 2;
        s.schedule = vec![
            ScheduleStep {
                duration_ms: 10,
                events: vec![0, 1],
            },
            ScheduleStep {
                duration_ms: 20,
                events: vec![2, 0],
            },
        ];

        let mut est = MockEstimator::new();
        est.add(0, 0, 1.5e-6, 0.9);
        est.add(0, 1, 3.0e-7, 0.2);

        let pb = s.build_update_prompt(&est);
        let response = s.client.chat(pb.build()).expect("LLM call should succeed");
        eprintln!("LLM update response:\n{response}");

        let steps = llm_common::parse_schedule_response(&response, &s.all_events, s.num_slots)
            .expect("should parse into a valid schedule");

        assert!(!steps.is_empty());
    }
}
