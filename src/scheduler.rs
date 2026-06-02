//! Pluggable counter-rotation policy: trait definition and decision type.

pub mod dynamic_llm;
pub mod fixed;
pub mod llm_common;
pub mod max_uncertainty;
pub mod random;
pub mod rate_of_change;
pub mod round_robin;
pub mod static_llm;
pub mod test;
pub mod weighted_round_robin_llm;

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::state::StateEstimator;
use std::time::Duration;

/// Pluggable counter selection policy.
pub trait Scheduler {
    /// Initialize with the universe of possible events and the number of hardware slots.
    fn init(
        &mut self,
        all_events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Return the next set of events to monitor, given the completed quantum and estimator state.
    ///
    /// The returned `active_events` must not exceed `num_slots` (as passed to `init`).
    fn next_step(&mut self, quantum: &Quantum, estimator: &dyn StateEstimator) -> ScheduleDecision;
}

/// Output from a scheduler step: which counters to activate and for how long.
pub struct ScheduleDecision {
    /// Events to activate for the next quantum; length must not exceed `num_slots`.
    pub active_events: Vec<EventId>,
    /// Overrides the default quantum duration when `Some`; uses the profiler default when `None`.
    pub duration: Option<Duration>,
}
