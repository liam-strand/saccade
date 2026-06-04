//! Rate-of-change scheduler: prioritizes events whose count rate is changing most
//! rapidly, using the triangle-area cost metric from Lim 2014.

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::scheduler::{ScheduleDecision, Scheduler};
use crate::state::StateEstimator;
use std::collections::{HashMap, VecDeque};

/// Scheduler that prioritizes hardware events with high rate-of-change (non-linearity).
///
/// Three scheduling tiers ensure all events are bootstrapped before cost-based selection:
/// - Tier 0 (never observed): always selected first, tiebroken by event_id ascending.
/// - Tier 1 (1–2 observations): selected before fully-bootstrapped events, tiebroken by
///   oldest-last-scheduled first.
/// - Tier 2 (≥3 observations): ranked by Lim 2014 triangle-area cost × δt.
pub struct RateOfChangeScheduler {
    events: Vec<EventId>,
    num_slots: usize,
    /// Last ≤3 (timestamp_ns, mean_rate) observations per event.
    history: HashMap<EventId, VecDeque<(u64, f64)>>,
    /// Timestamp (ns) when event was last placed in active_events; 0 = never.
    last_scheduled_ns: HashMap<EventId, u64>,
}

impl Default for RateOfChangeScheduler {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            num_slots: 6,
            history: HashMap::new(),
            last_scheduled_ns: HashMap::new(),
        }
    }
}

/// Lim 2014 triangle-area cost: deviation of observation B from the linear
/// interpolation of A→C, scaled by δt (nanoseconds since the event was last scheduled).
fn triangle_cost(a: (u64, f64), b: (u64, f64), c: (u64, f64), delta_t_ns: f64) -> f64 {
    let (ax, ay) = (a.0 as f64, a.1);
    let (bx, by) = (b.0 as f64, b.1);
    let (cx, cy) = (c.0 as f64, c.1);
    let delta_y = if cx != ax {
        (cy - ay) / (cx - ax) * (bx - ax)
    } else {
        0.0
    };
    ((by - ay - delta_y).abs() / 2.0) * delta_t_ns
}

impl Scheduler for RateOfChangeScheduler {
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
        quantum: &Quantum,
        _estimator: &dyn StateEstimator,
    ) -> ScheduleDecision {
        let now_ns = quantum.timestamp_ns();

        // Evict history entries older than 200× the quantum duration to avoid mixing
        // observations from different execution phases across long idle gaps.
        // Skipped when elapsed_ns is zero (e.g. test fixtures) to prevent spurious eviction.
        let stale_threshold = 200u64.saturating_mul(quantum.elapsed_ns());
        if stale_threshold > 0 {
            for h in self.history.values_mut() {
                while h
                    .front()
                    .is_some_and(|&(ts, _)| now_ns.saturating_sub(ts) > stale_threshold)
                {
                    h.pop_front();
                }
            }
        }

        // Record new observations; keep at most 3 entries (sliding window).
        for (&event_id, agg) in quantum.aggregates() {
            let h = self.history.entry(event_id).or_default();
            h.push_back((now_ns, agg.mean_rate));
            if h.len() > 3 {
                h.pop_front();
            }
        }

        // Assign each event a (cost, tiebreaker) pair, then sort descending.
        //
        // Tier 0: cost = f64::MAX,       tiebreaker encodes event_id ascending
        // Tier 1: cost = f64::MAX / 2,   tiebreaker encodes last_scheduled_ns ascending
        // Tier 2: cost = triangle_cost,  tiebreaker unused (0)
        let mut scored: Vec<(EventId, f64, u64)> = self
            .events
            .iter()
            .map(|&ev| {
                let hist_len = self.history.get(&ev).map_or(0, |h| h.len());
                let last_ns = self.last_scheduled_ns.get(&ev).copied().unwrap_or(0);
                let delta_t_ns = now_ns.saturating_sub(last_ns) as f64;

                let (cost, tiebreaker) = match hist_len {
                    0 => (f64::MAX, u64::MAX.wrapping_sub(ev as u64)),
                    1 | 2 => (f64::MAX / 2.0, u64::MAX.wrapping_sub(last_ns)),
                    _ => {
                        let h = self.history.get(&ev).unwrap();
                        let pts: Vec<(u64, f64)> = h.iter().copied().collect();
                        (triangle_cost(pts[0], pts[1], pts[2], delta_t_ns), 0u64)
                    }
                };
                (ev, cost, tiebreaker)
            })
            .collect();

        scored.sort_by(|(_, c1, t1), (_, c2, t2)| c2.total_cmp(c1).then(t2.cmp(t1)));

        let active_events: Vec<EventId> = scored
            .into_iter()
            .take(self.num_slots)
            .map(|(ev, _, _)| ev)
            .collect();

        for &ev in &active_events {
            self.last_scheduled_ns.insert(ev, now_ns);
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
    use crate::sample::{RawSample, TASK_COMM_LEN};
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

    /// Build a Quantum with one RawSample per (event_id, mean_rate) pair.
    fn make_quantum(event_rates: &[(EventId, f64)], timestamp_ns: u64, elapsed_ns: u64) -> Quantum {
        const DURATION_NS: u64 = 1_000_000;
        let samples: Vec<RawSample> = event_rates
            .iter()
            .map(|&(event_id, rate)| RawSample {
                timestamp_ns,
                duration_ns: DURATION_NS,
                cpu_id: 0,
                pid: 1,
                tid: 1,
                event_id,
                count: (rate * DURATION_NS as f64) as u64,
                task: [0u8; TASK_COMM_LEN],
            })
            .collect();
        Quantum::new(samples, timestamp_ns, elapsed_ns)
    }

    // Large enough elapsed_ns to prevent stale eviction across test timestamps.
    const ELAPSED: u64 = 10_000_000;

    #[test]
    fn triangle_cost_zero_for_collinear_points() {
        // A=(0,1), B=(1M,2), C=(2M,3): perfectly linear, cost should be ~0
        let cost = triangle_cost((0, 1.0), (1_000_000, 2.0), (2_000_000, 3.0), 1.0);
        assert!(cost.abs() < 1e-9, "cost={cost}");
    }

    #[test]
    fn triangle_cost_nonzero_for_spike() {
        // A=(0,1), B=(1M,10), C=(2M,2): spike at B → large triangle height
        let cost = triangle_cost((0, 1.0), (1_000_000, 10.0), (2_000_000, 2.0), 1.0);
        assert!(cost > 0.0, "cost={cost}");
    }

    #[test]
    fn tier0_events_selected_before_tier1() {
        let mut sched = RateOfChangeScheduler::default();
        sched.init(vec![1, 2, 3], 1).unwrap();
        let est = MockEstimator::new();

        // Observe only event 3 → it becomes tier 1; events 1 & 2 remain tier 0
        let d = sched.next_step(&make_quantum(&[(3, 1.0)], 1_000_000, ELAPSED), &est);
        // f64::MAX (tier 0) > f64::MAX/2 (tier 1); event 1 wins tier-0 tiebreak (lower id)
        assert_eq!(d.active_events, vec![1]);
    }

    #[test]
    fn tier1_favors_longest_unseen() {
        let mut sched = RateOfChangeScheduler::default();
        sched.init(vec![1, 2], 1).unwrap();
        let est = MockEstimator::new();

        // Step 1: both tier 0 → event 1 wins (lower id tiebreaker)
        let d1 = sched.next_step(
            &make_quantum(&[(1, 1.0), (2, 1.0)], 1_000_000, ELAPSED),
            &est,
        );
        assert_eq!(d1.active_events, vec![1]);

        // Step 2: both tier 1
        // Event 1: last_scheduled_ns = 1_000_000 → tiebreaker = u64::MAX - 1_000_000
        // Event 2: last_scheduled_ns = 0 (never) → tiebreaker = u64::MAX (higher → wins)
        let d2 = sched.next_step(
            &make_quantum(&[(1, 2.0), (2, 2.0)], 2_000_000, ELAPSED),
            &est,
        );
        assert_eq!(d2.active_events, vec![2]);
    }

    #[test]
    fn nonlinear_event_wins_over_linear_in_tier2() {
        let mut sched = RateOfChangeScheduler::default();
        sched.init(vec![1, 2], 1).unwrap();
        let est = MockEstimator::new();

        // Bootstrap: 3 quanta observed by both events.
        // Event 1: rates 1→2→3 (linear, triangle cost ≈ 0)
        // Event 2: rates 1→10→2 (spike at middle → high triangle cost)
        sched.next_step(&make_quantum(&[(1, 1.0), (2, 1.0)], 0, ELAPSED), &est);
        sched.next_step(
            &make_quantum(&[(1, 2.0), (2, 10.0)], 1_000_000, ELAPSED),
            &est,
        );
        let d = sched.next_step(
            &make_quantum(&[(1, 3.0), (2, 2.0)], 2_000_000, ELAPSED),
            &est,
        );

        assert_eq!(d.active_events, vec![2]);
    }

    #[test]
    fn selects_exactly_num_slots() {
        let mut sched = RateOfChangeScheduler::default();
        sched.init(vec![1, 2, 3, 4, 5], 3).unwrap();
        let est = MockEstimator::new();

        let d = sched.next_step(&make_quantum(&[], 1_000_000, ELAPSED), &est);
        assert_eq!(d.active_events.len(), 3);
    }

    #[test]
    fn fewer_events_than_slots_returns_all() {
        let mut sched = RateOfChangeScheduler::default();
        sched.init(vec![1, 2], 5).unwrap();
        let est = MockEstimator::new();

        let d = sched.next_step(&make_quantum(&[], 1_000_000, ELAPSED), &est);
        assert_eq!(d.active_events.len(), 2);
    }

    #[test]
    fn duration_is_none() {
        let mut sched = RateOfChangeScheduler::default();
        sched.init(vec![1], 4).unwrap();
        let est = MockEstimator::new();

        let d = sched.next_step(&make_quantum(&[(1, 1.0)], 1_000_000, ELAPSED), &est);
        assert!(d.duration.is_none());
    }
}
