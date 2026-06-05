//! Uncertainty-driven scheduler: prioritizes events with the highest mean estimation uncertainty.

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;
use std::collections::HashMap;
use rand::{prelude::SliceRandom, rng};


/// Selects the `num_slots` events whose per-thread uncertainty is highest on average, maximizing information gain.
pub struct MaxUncertaintyScheduler {
    /// Full set of candidate events, set by `init`.
    events: Vec<EventId>,
    /// Number of counters to activate per step, overridden by `init`.
    num_slots: usize,
}

impl MaxUncertaintyScheduler {
    /// Creates a scheduler with an empty event list and a default slot count of 6.
    fn new() -> Self {
        Self {
            events: Vec::new(),
            num_slots: 6,
        }
    }
}

impl Default for MaxUncertaintyScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler for MaxUncertaintyScheduler {
    fn init(
        &mut self,
        mut all_events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut r = rng();
        all_events.shuffle(&mut r);
        self.events = all_events;
        self.num_slots = num_slots;
        Ok(())
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        // Mean uncertainty per event across threads, for events the estimator
        // has already observed.
        let mean_uncertainty: HashMap<u32, f64> = estimator
            .all_estimates()
            .iter()
            .map(|((_tid, event_id), v)| (*event_id, v))
            .fold(
                HashMap::new(),
                |mut acc: HashMap<u32, Vec<f64>>, (id, v)| {
                    acc.entry(id).or_default().push(v.uncertainty);
                    acc
                },
            )
            .into_iter()
            .map(|(k, v)| {
                let sum: f64 = v.iter().sum();
                (k, sum / v.len() as f64)
            })
            .collect();

        // Rank the full candidate universe — not just observed events — so that
        // never-measured events (treated as maximally uncertain) are scheduled
        // first. Iterating only `all_estimates()` would deadlock at cold-start:
        // with no estimates yet, nothing is selected, nothing gets measured, and
        // the estimate map stays empty forever.
        let mut scored: Vec<(EventId, f64)> = self
            .events
            .iter()
            .map(|&ev| {
                (
                    ev,
                    mean_uncertainty.get(&ev).copied().unwrap_or(f64::INFINITY),
                )
            })
            .collect();

        // Sort by uncertainty descending, tiebroken by event_id ascending so
        // cold-start selection (all ties at +inf) is deterministic across seeds.
        scored.sort_by(|(id1, u1), (id2, u2)| u2.total_cmp(u1));

        ScheduleDecision {
            active_events: scored
                .into_iter()
                .take(self.num_slots)
                .map(|(id, _u)| id)
                .collect(),
            duration: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{CounterEstimate, EstimateKey};

    /// Minimal `StateEstimator` that stores a fixed map of uncertainty values for testing.
    struct MockEstimator {
        /// Stored estimates keyed by `(tid, event_id)`.
        estimates: HashMap<EstimateKey, CounterEstimate>,
    }

    impl MockEstimator {
        /// Creates an estimator with no entries.
        fn new() -> Self {
            Self {
                estimates: HashMap::new(),
            }
        }

        /// Inserts a `CounterEstimate` with the given `uncertainty` for `(tid, event_id)`.
        fn add(&mut self, tid: u32, event_id: EventId, uncertainty: f64) {
            self.estimates.insert(
                (tid, event_id),
                CounterEstimate {
                    uncertainty,
                    ..Default::default()
                },
            );
        }
    }

    impl StateEstimator for MockEstimator {
        fn measurement_update(
            &mut self,
            _tid: u32,
            _event_id: EventId,
            _rate: f64,
            _stddev: f64,
            _num_samples: u32,
            _timestamp_ns: u64,
        ) {
        }
        fn time_update(&mut self, _tid: u32, _event_id: EventId, _elapsed_ns: u64) {}
        fn rate(&self, _tid: u32, _event_id: EventId) -> f64 {
            0.0
        }
        fn uncertainty(&self, _tid: u32, _event_id: EventId) -> f64 {
            0.0
        }
        fn all_estimates(&self) -> &HashMap<EstimateKey, CounterEstimate> {
            &self.estimates
        }
    }

    /// Returns an empty quantum suitable for passing to schedulers under test.
    fn empty_quantum() -> Quantum {
        Quantum::new(vec![], 0, 0)
    }

    #[test]
    fn selects_highest_uncertainty_events() {
        let mut sched = MaxUncertaintyScheduler::default();
        sched.init(vec![1, 2, 3, 4, 5], 2).unwrap();

        let mut est = MockEstimator::new();
        est.add(1, 1, 0.1);
        est.add(1, 2, 0.9); // highest
        est.add(1, 3, 0.5);
        est.add(1, 4, 0.8); // second highest
        est.add(1, 5, 0.2);

        let decision = sched.next_step(&empty_quantum(), &est);

        let mut active = decision.active_events.clone();
        active.sort_unstable();
        assert_eq!(active, vec![2, 4]);
    }

    #[test]
    fn averages_uncertainty_across_threads() {
        let mut sched = MaxUncertaintyScheduler::default();
        sched.init(vec![1, 2], 1).unwrap();

        let mut est = MockEstimator::new();
        // event 1: avg = (0.2 + 0.4) / 2 = 0.3
        est.add(1, 1, 0.2);
        est.add(2, 1, 0.4);
        // event 2: avg = (0.8 + 0.6) / 2 = 0.7  — should win
        est.add(1, 2, 0.8);
        est.add(2, 2, 0.6);

        let decision = sched.next_step(&empty_quantum(), &est);

        assert_eq!(decision.active_events, vec![2]);
    }

    #[test]
    fn cold_start_bootstraps_all_events() {
        // With no estimates yet, every event is maximally uncertain, so the
        // scheduler must still activate candidates to bootstrap measurement
        // rather than returning nothing (which would deadlock the sampler).
        let mut sched = MaxUncertaintyScheduler::default();
        sched.init(vec![1, 2, 3], 4).unwrap();

        let decision = sched.next_step(&empty_quantum(), &MockEstimator::new());

        let mut active = decision.active_events.clone();
        active.sort_unstable();
        assert_eq!(active, vec![1, 2, 3]);
    }

    #[test]
    fn unmeasured_events_outrank_measured_ones() {
        // Event 2 has a recorded (finite) uncertainty; events 1 and 3 are
        // unmeasured (treated as +inf) and must be scheduled ahead of it.
        let mut sched = MaxUncertaintyScheduler::default();
        sched.init(vec![1, 2, 3], 2).unwrap();

        let mut est = MockEstimator::new();
        est.add(1, 2, 0.9);

        let decision = sched.next_step(&empty_quantum(), &est);

        let mut active = decision.active_events.clone();
        active.sort_unstable();
        assert_eq!(active, vec![1, 3]);
    }

    #[test]
    fn fewer_events_than_slots_returns_all() {
        let mut sched = MaxUncertaintyScheduler::default();
        sched.init(vec![1, 2], 4).unwrap();

        let mut est = MockEstimator::new();
        est.add(1, 1, 0.5);
        est.add(1, 2, 0.3);

        let decision = sched.next_step(&empty_quantum(), &est);

        assert_eq!(decision.active_events.len(), 2);
    }

    #[test]
    fn duration_is_none() {
        let mut sched = MaxUncertaintyScheduler::default();
        sched.init(vec![1], 4).unwrap();

        let mut est = MockEstimator::new();
        est.add(1, 1, 0.5);

        let decision = sched.next_step(&empty_quantum(), &est);

        assert!(decision.duration.is_none());
    }
}
