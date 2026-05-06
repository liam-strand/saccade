use crate::event::EventId;
use crate::llm::{LlmClient, PromptBuilder};
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;
use std::collections::HashSet;
use std::time::Duration;

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
        PromptBuilder::new()
            .system("You are a hardware performance counter scheduler for a Linux profiler.")
            .user(format!(
                "Available hardware performance counters:\n{event_list}\n\
                 Generate a cyclic observation schedule. Each step activates up to {num_slots} \
                 counters simultaneously and specifies how long to observe them in milliseconds.\n\n\
                 Respond with ONLY a JSON array — no prose, no markdown fences:\n\
                 [\n  {{ \"duration_ms\": <ms>, \"events\": [<id>, ...] }},\n  ...\n]\n\n\
                 Prefer counters that reveal bottlenecks: cache misses, branch mispredictions, \
                 TLB misses, memory stalls. Cover all provided counters across the full cycle. \
                 Each step: 1\u{2013}{num_slots} events.",
            ))
    }

    fn parse_schedule(&self, response: &str) -> Option<Vec<ScheduleStep>> {
        let start = response.find('[')?;
        let end = response.rfind(']')?;
        if end < start {
            return None;
        }
        let json_slice = &response[start..=end];

        #[derive(serde::Deserialize)]
        struct RawStep {
            duration_ms: u64,
            events: Vec<u32>,
        }

        let raw: Vec<RawStep> = serde_json::from_str(json_slice).ok()?;
        if raw.is_empty() {
            return None;
        }

        let valid_ids: HashSet<u32> = self.all_events.iter().copied().collect();
        let steps = raw
            .into_iter()
            .map(|s| {
                let events = s
                    .events
                    .into_iter()
                    .filter(|id| valid_ids.contains(id))
                    .take(self.num_slots)
                    .collect();
                ScheduleStep {
                    duration_ms: s.duration_ms.max(1),
                    events,
                }
            })
            .collect();
        Some(steps)
    }

    fn fallback_schedule(&self) -> Vec<ScheduleStep> {
        self.all_events
            .chunks(self.num_slots.max(1))
            .map(|chunk| ScheduleStep {
                duration_ms: 10,
                events: chunk.to_vec(),
            })
            .collect()
    }
}

impl Scheduler for StaticLlmScheduler {
    fn init(&mut self, all_events: Vec<EventId>, num_slots: usize) {
        self.all_events = all_events;
        self.num_slots = num_slots;

        let llm_result = {
            let pb = self.build_prompt();
            self.client.chat(pb.build())
        };

        let schedule = match llm_result {
            Ok(resp) => {
                if let Some(s) = self.parse_schedule(&resp) {
                    s
                } else {
                    tracing::warn!("Could not parse LLM schedule response; using fallback");
                    self.fallback_schedule()
                }
            }
            Err(e) => {
                tracing::warn!("LLM call failed ({e}); using fallback schedule");
                self.fallback_schedule()
            }
        };
        tracing::info!("StaticLlmScheduler: {} steps in cycle", schedule.len());
        self.schedule = schedule;
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        _estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        if self.schedule.is_empty() {
            return ScheduleDecision {
                active_events: self
                    .all_events
                    .iter()
                    .take(self.num_slots)
                    .cloned()
                    .collect(),
                duration: None,
            };
        }
        let step = &self.schedule[self.step_idx];
        let result = ScheduleDecision {
            active_events: step.events.clone(),
            duration: Some(Duration::from_millis(step.duration_ms)),
        };
        self.step_idx = (self.step_idx + 1) % self.schedule.len();
        result
    }
}
