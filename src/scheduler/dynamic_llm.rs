//! Scheduler that starts with an LLM-generated counter schedule and periodically refreshes it
//! in the background using live counter rate and uncertainty observations.

use crate::event::EventId;
use crate::llm::{ChatMessage, LlmClient, PromptBuilder};
use crate::quantum::Quantum;
use crate::scheduler::llm_common::{self, ScheduleStep};
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

/// LLM-based scheduler that adapts its counter rotation schedule at runtime using profiler feedback.
pub struct DynamicLlmScheduler {
    /// HTTP client used to call the LLM.
    client: LlmClient,
    /// Metadata for every available hardware event: `(id, name, description)`.
    event_info: Vec<(EventId, String, String)>,
    /// The currently active cyclic schedule.
    schedule: Vec<ScheduleStep>,
    /// Index of the next step to serve from `schedule`.
    step_idx: usize,
    /// All valid event IDs as reported by `init`.
    all_events: Vec<EventId>,
    /// Number of hardware counter slots the profiler can activate simultaneously.
    num_slots: usize,
    /// Number of `next_step` calls between background schedule refresh requests.
    update_interval: u32,
    /// Counter tracking calls since the last refresh request was sent.
    steps_since_update: u32,
    /// Channel to send a new prompt to the background worker thread.
    request_tx: Option<mpsc::SyncSender<Vec<ChatMessage>>>,
    /// Channel on which the background worker delivers the updated schedule.
    result_rx: Option<mpsc::Receiver<Result<Vec<ScheduleStep>, String>>>,
    /// Optional natural-language guidance forwarded to the LLM system message.
    guidance: Option<String>,
    /// When `true`, block the simulation loop during LLM calls to avoid racing ahead.
    simulation_mode: bool,
    /// Total number of `next_step` calls since `init`.
    step_count: u64,
    /// `step_count` value when the last query was dispatched (simulation mode only).
    dispatch_step: u64,
    /// Wall-clock time when the last query was dispatched (simulation mode only).
    dispatch_instant: Option<std::time::Instant>,
    /// `true` while the simulation loop should block waiting for a background response.
    waiting_for_response: bool,
    /// A received schedule held until `buffered_release_step` is reached.
    buffered_schedule: Option<Vec<ScheduleStep>>,
    /// The `step_count` at which `buffered_schedule` should be applied.
    buffered_release_step: u64,
}

impl DynamicLlmScheduler {
    /// Create a new scheduler; `init` must be called before `next_step` to populate the schedule.
    /// Set `simulation_mode` to `true` when replaying a trace (no real-time sleep between quanta)
    /// so that the scheduler blocks during LLM calls and replays the realistic delay afterwards.
    pub fn new(
        event_info: Vec<(EventId, String, String)>,
        client: LlmClient,
        update_interval: u32,
        guidance: Option<String>,
        simulation_mode: bool,
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
            guidance,
            simulation_mode,
            step_count: 0,
            dispatch_step: 0,
            dispatch_instant: None,
            waiting_for_response: false,
            buffered_schedule: None,
            buffered_release_step: 0,
        }
    }

    /// Build a prompt that includes current per-event rate and uncertainty observations so the LLM
    /// can reprioritize counters with high uncertainty or high event rate.
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

        let system = llm_common::system_message(num_slots, self.guidance.as_deref());
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
    /// Generate the initial schedule via LLM and spawn the background worker thread for updates.
    fn init(
        &mut self,
        all_events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.update_interval == 0 {
            return Err(
                format!("update_interval must be >= 1, got {}", self.update_interval).into(),
            );
        }
        self.all_events = all_events;
        self.num_slots = num_slots;

        self.schedule = {
            let pb = llm_common::build_init_prompt(
                &self.event_info,
                num_slots,
                self.guidance.as_deref(),
            );
            tracing::debug!("DynamicLlm init system message:\n{}", pb.build()[0].content);
            let messages = pb.build().to_vec();
            let client = &self.client;
            let all_events = &self.all_events;
            llm_common::chat_with_retry(
                |m| client.chat(m),
                messages,
                |resp| llm_common::parse_schedule_response(resp, all_events, num_slots),
                2,
            )
        }?;

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
                let result = llm_common::chat_with_retry(
                    |m| client.chat(m),
                    messages,
                    |resp| llm_common::parse_schedule_response(resp, &all_events, num_slots),
                    2,
                )
                .map_err(|e| e.to_string());
                if res_tx.send(result).is_err() {
                    break;
                }
            }
        });

        self.request_tx = Some(req_tx);
        self.result_rx = Some(res_rx);
        Ok(())
    }

    /// Serve the next scheduled step.
    ///
    /// In production mode this polls the background worker non-blockingly.
    /// In simulation mode this blocks during the LLM call, then buffers the result and releases
    /// it K quanta later (K = wall-clock LLM latency / quantum duration) to mirror production.
    fn next_step(
        &mut self,
        quantum: &Quantum,
        estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        // 0. Advance the step counter.
        self.step_count += 1;

        // 1. Apply any buffered schedule that has reached its release step.
        if self.buffered_schedule.is_some() && self.step_count >= self.buffered_release_step {
            self.schedule = self.buffered_schedule.take().unwrap();
            self.step_idx = 0;
            tracing::info!(
                "DynamicLlmScheduler: applying buffered update at step {}",
                self.step_count
            );
        }

        // 2. Poll or block for a completed background update.
        if self.simulation_mode && self.waiting_for_response {
            tracing::info!(
                "DynamicLlmScheduler: blocking for LLM response at step {}",
                self.step_count
            );
            if let Some(rx) = &self.result_rx {
                match rx.recv() {
                    Ok(Ok(steps)) => {
                        let w = self
                            .dispatch_instant
                            .map(|t| t.elapsed())
                            .unwrap_or_default();
                        let k = (w.as_nanos() / quantum.elapsed_ns().max(1) as u128).max(1) as u64;
                        tracing::info!(
                            "DynamicLlmScheduler: buffering update for release at step {} (K={})",
                            self.dispatch_step + k + 1,
                            k
                        );
                        self.buffered_schedule = Some(steps);
                        self.buffered_release_step = self.dispatch_step + k + 1;
                        self.waiting_for_response = false;
                    }
                    Ok(Err(msg)) => {
                        tracing::warn!(
                            "DynamicLlmScheduler: update failed, keeping schedule: {msg}"
                        );
                        self.waiting_for_response = false;
                    }
                    Err(_) => {
                        tracing::warn!("DynamicLlmScheduler: worker thread exited");
                        self.waiting_for_response = false;
                    }
                }
            }
        } else if let Some(rx) = &self.result_rx {
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

        // 3. Dispatch a new update request if the interval has elapsed.
        // Reset the counter only on successful send; if the worker is busy,
        // leave the counter alone so we retry on the next step.
        self.steps_since_update += 1;
        if self.steps_since_update >= self.update_interval {
            let messages = self.build_update_prompt(estimator).build().to_vec();
            if let Some(tx) = &self.request_tx
                && tx.try_send(messages).is_ok()
            {
                self.steps_since_update = 0;
                if self.simulation_mode {
                    self.dispatch_step = self.step_count;
                    self.dispatch_instant = Some(std::time::Instant::now());
                    self.waiting_for_response = true;
                }
            }
        }

        // 4. Serve the current schedule step.
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

    /// A `StateEstimator` that holds a fixed set of manually inserted observations.
    struct MockEstimator {
        /// Observations keyed by `(thread_id, event_id)`.
        estimates: HashMap<EstimateKey, CounterEstimate>,
    }

    impl MockEstimator {
        /// Create an estimator with no observations.
        fn new() -> Self {
            Self {
                estimates: HashMap::new(),
            }
        }

        /// Insert a synthetic observation for the given thread and event.
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

    /// Produce a `Quantum` with no samples, suitable as a placeholder argument in scheduler tests.
    fn empty_quantum() -> Quantum {
        Quantum::new(vec![], 0, 0)
    }

    /// Build a `DynamicLlmScheduler` pre-populated with a two-step schedule and three events.
    fn make_scheduler(update_interval: u32, simulation_mode: bool) -> DynamicLlmScheduler {
        let mut s = DynamicLlmScheduler::new(
            vec![
                (0, "cache-misses".into(), "Cache misses".into()),
                (1, "branch-misses".into(), "Branch mispredictions".into()),
                (2, "instructions".into(), "Instructions retired".into()),
            ],
            LlmClient::new("http://localhost:0", "test-model"),
            update_interval,
            None,
            simulation_mode,
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
        let mut s = make_scheduler(100, false);
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
        let s = make_scheduler(100, false);
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
    fn init_rejects_zero_update_interval() {
        let mut s = DynamicLlmScheduler::new(
            vec![],
            LlmClient::new("http://localhost:0", "test-model"),
            0,
            None,
            false,
        );
        let result = s.init(vec![], 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("update_interval"));
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
            None,
            false,
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

    #[test]
    fn buffered_update_applied_at_correct_step() {
        let mut s = make_scheduler(100, true);
        let q = empty_quantum();
        let est = MockEstimator::new();

        // Pre-populate a buffered schedule distinct from the current one.
        s.buffered_schedule = Some(vec![ScheduleStep {
            duration_ms: 5,
            events: vec![1, 2],
        }]);
        s.buffered_release_step = 5;

        // Steps 1–4: buffer should still be held; events served from original schedule.
        for _ in 0..4 {
            s.next_step(&q, &est);
            assert!(
                s.buffered_schedule.is_some(),
                "buffer should not be released before step 5"
            );
        }

        // Step 5 (step_count == 5): buffer released; new schedule applied.
        let d5 = s.next_step(&q, &est);
        assert!(
            s.buffered_schedule.is_none(),
            "buffer should be consumed at release step"
        );
        assert_eq!(
            d5.active_events,
            vec![1, 2],
            "first decision after release should come from buffered schedule"
        );
    }
}
