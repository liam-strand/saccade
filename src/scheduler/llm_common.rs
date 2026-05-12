use crate::event::EventId;
use crate::llm::PromptBuilder;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct ScheduleStep {
    pub(super) duration_ms: u64,
    pub(super) events: Vec<EventId>,
}

#[derive(Debug, Deserialize)]
pub(super) struct EventWeight {
    pub(super) event_id: EventId,
    pub(super) weight: u32,
}

pub(super) fn system_message(num_slots: usize, guidance: Option<&str>) -> String {
    let guidance_suffix = guidance
        .map(|g| format!("\n\nUser guidance: {g}"))
        .unwrap_or_default();
    format!(
        "You are an expert Linux performance profiling assistant. \
         You generate hardware performance counter observation schedules \
         for a sampling profiler that activates {num_slots} \
         counters simultaneously. The profiler cycles through the schedule \
         indefinitely, rotating which counters are active to build a \
         complete picture of program behavior over time.{guidance_suffix}"
    )
}

pub(super) fn build_init_prompt(
    event_info: &[(EventId, String, String)],
    num_slots: usize,
    guidance: Option<&str>,
) -> PromptBuilder {
    let mut event_list = String::new();
    for (id, name, desc) in event_info {
        event_list.push_str(&format!("  {id}: {name} — {desc}\n"));
    }
    let ids: Vec<EventId> = event_info.iter().map(|(id, _, _)| *id).collect();
    let durations = [25u64, 75, 50];
    let example_steps: Vec<ScheduleStep> = ids
        .chunks(num_slots.max(1))
        .take(3)
        .enumerate()
        .map(|(i, chunk)| ScheduleStep {
            duration_ms: durations[i % durations.len()],
            events: chunk.to_vec(),
        })
        .collect();
    let example = serde_json::to_string_pretty(&example_steps).expect("should serialize");

    let system = system_message(num_slots, guidance);
    let user = format!(
        "\
Available performance counters (format — ID: name: description):
{event_list}
Generate a cyclic measurement schedule that covers all of the \
counters above. Each step activates {num_slots} counters \
and runs for a specified duration (aim for 10–1000 ms per step). \
Prioritize counters that reveal common bottlenecks — cache misses, \
TLB pressure, branch mispredictions, memory stalls — and ensure \
every counter appears at least once across the full cycle.

Example output (hypothetical IDs):
{example}

Produce the schedule for the counters listed above. \
Your entire response must be a single JSON array. \
No explanation, no prose, no markdown fences."
    );
    PromptBuilder::new().system(system).user(user)
}

pub(super) fn parse_schedule_response(
    response: &str,
    all_events: &[EventId],
    num_slots: usize,
) -> Result<Vec<ScheduleStep>, String> {
    let start = response
        .find('[')
        .ok_or_else(|| format!("no JSON array found in LLM response:\n{response}"))?;
    let end = response
        .rfind(']')
        .ok_or_else(|| format!("JSON array is not closed in LLM response:\n{response}"))?;
    if end < start {
        return Err(format!("malformed JSON array in LLM response:\n{response}"));
    }

    let steps: Vec<ScheduleStep> = serde_json::from_str(&response[start..=end])
        .map_err(|e| format!("failed to parse LLM schedule JSON: {e}\nresponse:\n{response}"))?;

    if steps.is_empty() {
        return Err("LLM returned an empty schedule".to_string());
    }

    let valid_ids: HashSet<u32> = all_events.iter().copied().collect();
    let steps: Vec<ScheduleStep> = steps
        .into_iter()
        .enumerate()
        .filter_map(|(i, mut step)| {
            step.events.retain(|id| valid_ids.contains(id));
            step.events.truncate(num_slots);
            step.duration_ms = step.duration_ms.max(1);
            if step.events.is_empty() {
                tracing::warn!("dropping step {i}: no valid event IDs after filtering");
                None
            } else {
                Some(step)
            }
        })
        .collect();

    if steps.is_empty() {
        return Err("all steps contained invalid event IDs".to_string());
    }

    Ok(steps)
}

pub(super) fn build_weights_prompt(
    event_info: &[(EventId, String, String)],
    num_slots: usize,
    guidance: Option<&str>,
) -> PromptBuilder {
    let mut event_list = String::new();
    for (id, name, desc) in event_info {
        event_list.push_str(&format!("  {id}: {name} — {desc}\n"));
    }
    let system = system_message(num_slots, guidance);
    let user = format!(
        "Available performance counters (format — ID: name: description):
{event_list}
Assign a priority weight (integer 1–10) to each counter. \
A higher weight means the counter should be sampled more often. \
Prioritize counters that reveal common bottlenecks — cache misses, \
TLB pressure, branch mispredictions, memory stalls. \
Every counter must appear exactly once in your response.

Your entire response must be a single JSON array of objects with \
\"event_id\" (integer) and \"weight\" (integer 1–10) fields. \
No explanation, no prose, no markdown fences."
    );
    PromptBuilder::new().system(system).user(user)
}

pub(super) fn parse_weights_response(
    response: &str,
    all_events: &[EventId],
) -> Result<Vec<EventWeight>, String> {
    let start = response
        .find('[')
        .ok_or_else(|| format!("no JSON array in response:\n{response}"))?;
    let end = response
        .rfind(']')
        .ok_or_else(|| format!("JSON array not closed:\n{response}"))?;
    if end < start {
        return Err(format!("malformed JSON array:\n{response}"));
    }

    let parsed: Vec<EventWeight> = serde_json::from_str(&response[start..=end])
        .map_err(|e| format!("failed to parse weights JSON: {e}\nresponse:\n{response}"))?;

    let valid_ids: HashSet<EventId> = all_events.iter().copied().collect();
    let mut by_id: HashMap<EventId, u32> = HashMap::new();
    for ew in parsed {
        if valid_ids.contains(&ew.event_id) {
            let w = ew.weight.clamp(1, 10);
            by_id
                .entry(ew.event_id)
                .and_modify(|v| *v = (*v).max(w))
                .or_insert(w);
        }
    }

    for &id in all_events {
        by_id.entry(id).or_insert(1);
    }

    if by_id.is_empty() {
        return Err("no valid event weights in response".to_string());
    }
    Ok(by_id
        .into_iter()
        .map(|(event_id, weight)| EventWeight { event_id, weight })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events() -> Vec<EventId> {
        vec![0, 1, 2, 3, 4]
    }

    #[test]
    fn system_message_includes_guidance() {
        let msg = system_message(4, Some("focus on memory bandwidth"));
        assert!(msg.contains("focus on memory bandwidth"));
    }

    #[test]
    fn system_message_no_guidance() {
        let msg = system_message(4, None);
        assert!(!msg.contains("User guidance"));
    }

    #[test]
    fn parse_valid_schedule() {
        let json = r#"[
            {"duration_ms": 25, "events": [0, 1]},
            {"duration_ms": 30, "events": [2, 3, 4]}
        ]"#;
        let steps = parse_schedule_response(json, &events(), 4).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].duration_ms, 25);
        assert_eq!(steps[0].events, vec![0, 1]);
        assert_eq!(steps[1].events, vec![2, 3, 4]);
    }

    #[test]
    fn parse_no_array() {
        let err = parse_schedule_response("the model just said hello", &events(), 4).unwrap_err();
        assert!(err.contains("no JSON array found"));
    }

    #[test]
    fn parse_empty_array() {
        let err = parse_schedule_response("[]", &events(), 4).unwrap_err();
        assert!(err.contains("empty"));
    }

    #[test]
    fn parse_filters_invalid_ids() {
        let json = r#"[{"duration_ms": 20, "events": [0, 99]}]"#;
        let steps = parse_schedule_response(json, &[0, 1], 4).unwrap();
        assert_eq!(steps[0].events, vec![0]);
    }

    #[test]
    fn parse_all_invalid_ids_errors() {
        let json = r#"[{"duration_ms": 20, "events": [99, 100]}]"#;
        let err = parse_schedule_response(json, &[0, 1], 4).unwrap_err();
        assert!(err.contains("invalid event IDs"), "got: {err}");
    }

    #[test]
    fn parse_skips_bad_step_keeps_good() {
        let json = r#"[
            {"duration_ms": 20, "events": [99, 100]},
            {"duration_ms": 30, "events": [0, 1]}
        ]"#;
        let steps = parse_schedule_response(json, &[0, 1], 4).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].events, vec![0, 1]);
    }

    #[test]
    fn parse_clamps_zero_duration() {
        let json = r#"[{"duration_ms": 0, "events": [0]}]"#;
        let steps = parse_schedule_response(json, &[0], 4).unwrap();
        assert_eq!(steps[0].duration_ms, 1);
    }

    #[test]
    fn parse_truncates_to_num_slots() {
        let json = r#"[{"duration_ms": 10, "events": [0, 1, 2, 3, 4]}]"#;
        let steps = parse_schedule_response(json, &events(), 2).unwrap();
        assert_eq!(steps[0].events.len(), 2);
    }

    #[test]
    fn parse_valid_weights() {
        let json = r#"[{"event_id": 0, "weight": 5}, {"event_id": 1, "weight": 2}]"#;
        let weights = parse_weights_response(json, &[0, 1]).unwrap();
        let map: HashMap<EventId, u32> =
            weights.into_iter().map(|ew| (ew.event_id, ew.weight)).collect();
        assert_eq!(map[&0], 5);
        assert_eq!(map[&1], 2);
    }

    #[test]
    fn parse_weights_fills_missing_events() {
        let json = r#"[{"event_id": 0, "weight": 5}]"#;
        let weights = parse_weights_response(json, &[0, 1, 2]).unwrap();
        let map: HashMap<EventId, u32> =
            weights.into_iter().map(|ew| (ew.event_id, ew.weight)).collect();
        assert_eq!(map[&0], 5);
        assert_eq!(map[&1], 1);
        assert_eq!(map[&2], 1);
    }

    #[test]
    fn parse_weights_clamps_out_of_range() {
        let json = r#"[{"event_id": 0, "weight": 0}, {"event_id": 1, "weight": 15}]"#;
        let weights = parse_weights_response(json, &[0, 1]).unwrap();
        let map: HashMap<EventId, u32> =
            weights.into_iter().map(|ew| (ew.event_id, ew.weight)).collect();
        assert_eq!(map[&0], 1);
        assert_eq!(map[&1], 10);
    }

    #[test]
    fn parse_weights_filters_unknown_ids() {
        let json = r#"[{"event_id": 0, "weight": 5}, {"event_id": 99, "weight": 8}]"#;
        let weights = parse_weights_response(json, &[0, 1]).unwrap();
        let map: HashMap<EventId, u32> =
            weights.into_iter().map(|ew| (ew.event_id, ew.weight)).collect();
        assert!(!map.contains_key(&99));
        assert_eq!(map[&0], 5);
        assert_eq!(map[&1], 1);
    }

    #[test]
    fn parse_weights_deduplicates_keeps_max() {
        let json = r#"[{"event_id": 0, "weight": 3}, {"event_id": 0, "weight": 7}]"#;
        let weights = parse_weights_response(json, &[0]).unwrap();
        let map: HashMap<EventId, u32> =
            weights.into_iter().map(|ew| (ew.event_id, ew.weight)).collect();
        assert_eq!(map[&0], 7);
    }

    #[test]
    fn parse_weights_no_array() {
        let err = parse_weights_response("hello world", &[0]).unwrap_err();
        assert!(err.contains("no JSON array"));
    }
}
