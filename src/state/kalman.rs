//! Kalman filter state estimator.
//!
//! Maintains a per-thread multivariate Gaussian over event rates with full
//! covariance. `measurement_update` is a scalar update (H = one-hot for the
//! observed event); process-noise inflation is tracked *per event* via
//! `last_predicted_ns[i]`, so only the observed event's diagonal is
//! advanced on measurement. `time_update` then advances a stale event's
//! diagonal independently, without double-counting.
//!
//! **Why per-event timing.** Under `F = I` and diagonal `Q`, the predict
//! step affects only diagonals of `P`. Off-diagonals are time-invariant,
//! so each event can carry its own "predicted-up-to" timestamp without
//! temporal incoherence in the covariance matrix. This matches the
//! profiler's call pattern (one `measurement_update` per observed event,
//! one `time_update` per stale event, each with the quantum's elapsed
//! time) with no cross-talk.
//!
//! **When this needs to change.** Introducing off-diagonal `Q` (for
//! cross-correlated process noise) breaks the "off-diagonals are
//! time-invariant" property. At that point `predict_event_to` must become
//! a full-filter predict keyed on a thread-level `last_predicted_ns`, and
//! the profiler contract needs to guarantee that `time_update` is not
//! called for a thread that already had a `measurement_update` in the
//! same step.
//!
//! **Why multivariate per-thread at all.** With diagonal `Q` and no
//! seeded off-diagonal `P`, scalar updates never propagate across events
//! and the filter reduces to independent scalar filters. The multivariate
//! matrix structure is in place so that:
//!   - off-diagonal `Q` terms can be seeded from a calibration run, or
//!   - an online estimator can populate them from observed rate
//! correlations as a drop-in extension.

use crate::event::EventId;
use crate::state::{CounterEstimate, EstimateKey, StateEstimator};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct KalmanConfig {
    pub process_noise_per_ns: f64,
    pub min_measurement_variance: f64,
    pub initial_variance: f64,
    pub uncertainty_reference_variance: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise_per_ns: 1e-18,
            min_measurement_variance: 1e-18,
            initial_variance: 1e-6,
            uncertainty_reference_variance: 1e-12,
        }
    }
}

#[derive(Default)]
struct ThreadFilter {
    event_index: HashMap<EventId, usize>,
    index_event: Vec<EventId>,
    x: Vec<f64>,
    p: Vec<Vec<f64>>,
    /// Timestamp of most recent `measurement_update` per event (for reporting).
    last_ts: Vec<u64>,
    /// Time up to which `P[i,i]` reflects accumulated process noise.
    /// Advanced by `predict_event_to`; never moves backward.
    last_predicted_ns: Vec<u64>,
    sample_counts: Vec<u64>,
    last_stddev: Vec<f64>,
}

impl ThreadFilter {
    fn ensure_event(&mut self, event_id: EventId, config: &KalmanConfig) -> usize {
        if let Some(&idx) = self.event_index.get(&event_id) {
            return idx;
        }
        let idx = self.index_event.len();
        self.event_index.insert(event_id, idx);
        self.index_event.push(event_id);
        self.x.push(0.0);
        self.last_ts.push(0);
        self.last_predicted_ns.push(0);
        self.sample_counts.push(0);
        self.last_stddev.push(0.0);
        for row in &mut self.p {
            row.push(0.0);
        }
        let mut new_row = vec![0.0; idx + 1];
        new_row[idx] = config.initial_variance;
        self.p.push(new_row);
        idx
    }

    /// Inflate `P[event_idx, event_idx]` by `Q · (target_ns − last_predicted_ns[event_idx])`.
    /// Idempotent; never moves the event's clock backward.
    ///
    /// NOTE: Only inflates the single event's diagonal. Correct under `F = I`
    /// and diagonal `Q`. For off-diagonal `Q`, this must become a full-filter
    /// predict keyed on a thread-level clock — see module docs.
    fn predict_event_to(&mut self, event_idx: usize, target_ns: u64, config: &KalmanConfig) {
        if target_ns > self.last_predicted_ns[event_idx] {
            let delta = target_ns - self.last_predicted_ns[event_idx];
            self.p[event_idx][event_idx] += config.process_noise_per_ns * delta as f64;
            self.last_predicted_ns[event_idx] = target_ns;
        }
    }

    fn scalar_update(&mut self, event_idx: usize, z: f64, r: f64) {
        let n = self.x.len();
        let s = self.p[event_idx][event_idx] + r;
        if !s.is_finite() || s <= 0.0 {
            return;
        }
        let k: Vec<f64> = (0..n).map(|j| self.p[j][event_idx] / s).collect();
        let y = z - self.x[event_idx];
        for j in 0..n {
            self.x[j] += k[j] * y;
        }
        let row = self.p[event_idx].clone();
        for j in 0..n {
            let kj = k[j];
            let pj = &mut self.p[j];
            for (kk, pval) in pj.iter_mut().enumerate() {
                *pval -= kj * row[kk];
            }
        }
        for j in 0..n {
            for kk in (j + 1)..n {
                let avg = 0.5 * (self.p[j][kk] + self.p[kk][j]);
                self.p[j][kk] = avg;
                self.p[kk][j] = avg;
            }
        }
    }
}

pub struct KalmanFilterEstimator {
    threads: HashMap<u32, ThreadFilter>,
    snapshots: HashMap<EstimateKey, CounterEstimate>,
    config: KalmanConfig,
}

impl KalmanFilterEstimator {
    pub fn new() -> Self {
        Self::with_config(KalmanConfig::default())
    }

    pub fn with_config(config: KalmanConfig) -> Self {
        Self {
            threads: HashMap::new(),
            snapshots: HashMap::new(),
            config,
        }
    }

    pub fn config(&self) -> &KalmanConfig {
        &self.config
    }

    fn uncertainty_from_variance(&self, var: f64) -> f64 {
        let v = var.max(0.0);
        let denom = v + self.config.uncertainty_reference_variance;
        if denom <= 0.0 {
            1.0
        } else {
            (v / denom).clamp(0.0, 1.0)
        }
    }

    fn refresh_snapshots(&mut self, tid: u32) {
        let Some(filter) = self.threads.get(&tid) else {
            return;
        };
        for (idx, &event_id) in filter.index_event.iter().enumerate() {
            let rate = filter.x[idx];
            let var = filter.p[idx][idx];
            let uncertainty = self.uncertainty_from_variance(var);
            self.snapshots.insert(
                (tid, event_id),
                CounterEstimate {
                    rate,
                    rate_stddev: filter.last_stddev[idx],
                    uncertainty,
                    last_updated_ns: filter.last_ts[idx],
                    sample_count: filter.sample_counts[idx],
                },
            );
        }
    }
}

impl Default for KalmanFilterEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl StateEstimator for KalmanFilterEstimator {
    fn measurement_update(
        &mut self,
        tid: u32,
        event_id: EventId,
        rate: f64,
        stddev: f64,
        num_samples: u32,
        timestamp_ns: u64,
    ) {
        let filter = self.threads.entry(tid).or_default();
        let idx = filter.ensure_event(event_id, &self.config);
        filter.predict_event_to(idx, timestamp_ns, &self.config);

        let n = num_samples.max(1) as f64;
        let r = ((stddev * stddev) / n).max(self.config.min_measurement_variance);

        filter.scalar_update(idx, rate, r);
        filter.last_ts[idx] = timestamp_ns;
        filter.sample_counts[idx] = filter.sample_counts[idx].saturating_add(num_samples as u64);
        filter.last_stddev[idx] = stddev;

        self.refresh_snapshots(tid);
    }

    fn time_update(&mut self, tid: u32, event_id: EventId, elapsed_ns: u64) {
        let filter = self.threads.entry(tid).or_default();
        let idx = filter.ensure_event(event_id, &self.config);
        let target = filter.last_predicted_ns[idx].saturating_add(elapsed_ns);
        filter.predict_event_to(idx, target, &self.config);
        self.refresh_snapshots(tid);
    }

    fn rate(&self, tid: u32, event_id: EventId) -> f64 {
        self.snapshots.get(&(tid, event_id)).map_or(0.0, |e| e.rate)
    }

    fn uncertainty(&self, tid: u32, event_id: EventId) -> f64 {
        self.snapshots
            .get(&(tid, event_id))
            .map_or(1.0, |e| e.uncertainty)
    }

    fn all_estimates(&self) -> &HashMap<EstimateKey, CounterEstimate> {
        &self.snapshots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_measurement_drops_uncertainty() {
        let mut kf = KalmanFilterEstimator::new();
        kf.measurement_update(1, 0, 1e-6, 1e-9, 100, 1_000);
        assert!((kf.rate(1, 0) - 1e-6).abs() < 1e-7);
        assert!(kf.uncertainty(1, 0) < 0.5);
    }

    #[test]
    fn waiting_grows_uncertainty() {
        let mut kf = KalmanFilterEstimator::new();
        kf.measurement_update(1, 0, 1e-6, 1e-9, 100, 1_000);
        let u0 = kf.uncertainty(1, 0);
        kf.time_update(1, 0, 1_000_000_000);
        assert!(kf.uncertainty(1, 0) > u0);
    }

    #[test]
    fn stale_event_advances_in_profiler_pattern() {
        // Simulates two quanta. In the second, event 0 is observed and
        // event 1 is stale — the profiler would call `measurement_update`
        // for 0 and `time_update` for 1. Event 1's uncertainty must grow.
        let mut kf = KalmanFilterEstimator::new();
        kf.measurement_update(1, 0, 1e-6, 1e-12, 100, 1_000);
        kf.measurement_update(1, 1, 2e-6, 1e-12, 100, 1_000);
        let u1_before = kf.uncertainty(1, 1);

        kf.measurement_update(1, 0, 1.1e-6, 1e-12, 100, 2_000);
        kf.time_update(1, 1, 1_000);

        assert!(kf.uncertainty(1, 1) > u1_before);
    }

    #[test]
    fn measurement_does_not_inflate_other_events() {
        // An observation of event 0 must not advance event 1's per-event
        // clock — otherwise a subsequent `time_update` for event 1 would
        // double-count the elapsed time.
        let mut kf = KalmanFilterEstimator::new();
        kf.measurement_update(1, 0, 1e-6, 1e-12, 100, 1_000);
        kf.measurement_update(1, 1, 2e-6, 1e-12, 100, 1_000);
        let filter = &kf.threads[&1];
        let idx1 = filter.event_index[&1];
        let clock_before = filter.last_predicted_ns[idx1];

        kf.measurement_update(1, 0, 1.1e-6, 1e-12, 100, 2_000);
        let clock_after = kf.threads[&1].last_predicted_ns[idx1];
        assert_eq!(clock_before, clock_after);
    }

    #[test]
    fn high_variance_measurement_gets_less_weight() {
        let mut kf_clean = KalmanFilterEstimator::new();
        let mut kf_noisy = KalmanFilterEstimator::new();
        kf_clean.measurement_update(1, 0, 1e-6, 1e-12, 100, 1_000);
        kf_noisy.measurement_update(1, 0, 1e-6, 1e-3, 100, 1_000);
        kf_clean.measurement_update(1, 0, 2e-6, 1e-12, 100, 2_000);
        kf_noisy.measurement_update(1, 0, 2e-6, 1e-3, 100, 2_000);
        assert!(kf_clean.rate(1, 0) > kf_noisy.rate(1, 0));
    }
}
