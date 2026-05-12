use crate::event::EventId;
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;
use std::collections::HashMap;

pub struct DistributionScheduler {
    events: Vec<EventId>,
    num_slots: usize,
}

impl DistributionScheduler {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            num_slots: 4,
        }
    }
}

impl Default for DistributionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler for DistributionScheduler {
    fn init(
        &mut self,
        all_events: Vec<EventId>,
        num_slots: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.events = all_events;
        self.num_slots = num_slots;
        Ok(())
    }

    fn next_step(
        &mut self,
        _quantum: &Quantum,
        estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        let mut res: Vec<_> = estimator
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

        res.sort_by(|(_k1, avg1), (_k2, avg2)| avg1.total_cmp(avg2));

        ScheduleDecision {
            active_events: res
                .into_iter()
                .rev()
                .take(self.num_slots)
                .map(|(id, _avg)| id)
                .collect(),
            duration: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{CounterEstimate, EstimateKey};

    struct MockEstimator {
        estimates: HashMap<EstimateKey, CounterEstimate>,
    }

    impl MockEstimator {
        fn new() -> Self {
            Self {
                estimates: HashMap::new(),
            }
        }

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

    fn empty_quantum() -> Quantum {
        Quantum::new(vec![], 0, 0)
    }

    #[test]
    fn selects_highest_uncertainty_events() {
        let mut sched = DistributionScheduler::default();
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
        let mut sched = DistributionScheduler::default();
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
    fn empty_estimator_returns_empty() {
        let mut sched = DistributionScheduler::default();
        sched.init(vec![1, 2, 3], 4).unwrap();

        let decision = sched.next_step(&empty_quantum(), &MockEstimator::new());

        assert!(decision.active_events.is_empty());
    }

    #[test]
    fn fewer_events_than_slots_returns_all() {
        let mut sched = DistributionScheduler::default();
        sched.init(vec![1, 2], 4).unwrap();

        let mut est = MockEstimator::new();
        est.add(1, 1, 0.5);
        est.add(1, 2, 0.3);

        let decision = sched.next_step(&empty_quantum(), &est);

        assert_eq!(decision.active_events.len(), 2);
    }

    #[test]
    fn duration_is_none() {
        let mut sched = DistributionScheduler::default();
        sched.init(vec![1], 4).unwrap();

        let mut est = MockEstimator::new();
        est.add(1, 1, 0.5);

        let decision = sched.next_step(&empty_quantum(), &est);

        assert!(decision.duration.is_none());
    }
}
