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
//! **Off-diagonal Q via `quantum_step`.** `KalmanFilterEstimator::quantum_step`
//! adds correlated process noise to off-diagonal `P` entries once per quantum.
//! The per-event diagonal clocks are unaffected — they continue to track when
//! each individual event's diagonal was last inflated. Off-diagonal Q is
//! parameterized as `process_noise_scale * process_noise_per_ns * elapsed_ns * r`
//! where `r` is the Pearson correlation from offline calibration, so its
//! magnitude matches the diagonal increment.
//!
//! **Why multivariate per-thread at all.** With diagonal `Q` and no
//! seeded off-diagonal `P`, scalar updates never propagate across events
//! and the filter reduces to independent scalar filters. The multivariate
//! matrix structure is in place so that:
//! - off-diagonal `Q` terms can be seeded from a calibration run, or
//! - an online estimator can populate them from observed rate
//!   correlations as a drop-in extension.

use crate::event::EventId;
use crate::state::{CounterEstimate, EstimateKey, StateEstimator};
use std::collections::HashMap;
use std::path::PathBuf;

/// Sparse off-diagonal correlation data loaded from a calibration JSON file.
///
/// Stores only pairs `(i, j)` with `i < j` and `|r| > 1e-9` so that
/// iteration over non-zero entries is O(non-zero) rather than O(n²).
#[derive(Debug, Default)]
pub struct CorrelationData {
    /// `correlations[(i, j)]` = Pearson r for event pair, `i < j`.
    correlations: HashMap<(usize, usize), f64>,
    /// Per-event rate variance [events/ns]², used to seed initial P.
    pub variances: Vec<f64>,
    /// Map from EventId to the index used in `variances` and the JSON.
    event_index: HashMap<EventId, usize>,
}

impl CorrelationData {
    /// Return the Pearson r for an event index pair, regardless of argument order; 0.0 if absent.
    fn get(&self, a: usize, b: usize) -> f64 {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        self.correlations.get(&(lo, hi)).copied().unwrap_or(0.0)
    }

    /// Map an `EventId` to its index in `variances` and the correlation matrix.
    fn idx(&self, event_id: EventId) -> Option<usize> {
        self.event_index.get(&event_id).copied()
    }
}

/// Configuration for the multivariate Kalman filter estimator.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KalmanConfig {
    /// Process noise added to `P[i,i]` per nanosecond of elapsed time (units: (events/ns)²/ns).
    pub process_noise_per_ns: f64,
    /// Floor applied to measurement variance `R` to prevent division by near-zero values.
    pub min_measurement_variance: f64,
    /// Initial diagonal value of `P` when a new event is introduced (units: (events/ns)²).
    pub initial_variance: f64,
    /// Reference variance used to normalize `P[i,i]` into a [0, 1] uncertainty score;
    /// a diagonal entry equal to this value maps to uncertainty ≈ 0.5.
    pub uncertainty_reference_variance: f64,
    /// Path to a `correlation.json` file produced by `python/correlation.py`.
    /// When set, off-diagonal P entries are seeded on event introduction
    /// and refreshed each quantum via `quantum_step`.
    #[serde(default)]
    pub correlation_path: Option<PathBuf>,
    /// Fraction of the diagonal Q increment applied to off-diagonal entries
    /// per quantum. `dQ_ij = process_noise_scale * process_noise_per_ns *
    /// elapsed_ns * r_ij`. Default: 0.1.
    #[serde(default = "KalmanConfig::default_correlation_process_noise_scale")]
    pub correlation_process_noise_scale: f64,
}

impl KalmanConfig {
    /// Returns the default `correlation_process_noise_scale` (0.1) for use with `#[serde(default = ...)]`.
    fn default_correlation_process_noise_scale() -> f64 {
        0.1
    }
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise_per_ns: 1e-18,
            min_measurement_variance: 1e-18,
            initial_variance: 1e-6,
            uncertainty_reference_variance: 1e-12,
            correlation_path: None,
            correlation_process_noise_scale: Self::default_correlation_process_noise_scale(),
        }
    }
}

/// Per-thread multivariate Kalman filter state.
#[derive(Default)]
struct ThreadFilter {
    /// Maps `EventId` to its column/row index in `x` and `p`.
    event_index: HashMap<EventId, usize>,
    /// Inverse of `event_index`; maps index back to `EventId`.
    index_event: Vec<EventId>,
    /// State vector: estimated rate (events/ns) per event.
    x: Vec<f64>,
    /// Covariance matrix stored as a row-major `Vec<Vec<f64>>`.
    p: Vec<Vec<f64>>,
    /// Timestamp (ns) of the most recent `measurement_update` per event; used only for reporting.
    last_ts: Vec<u64>,
    /// Time up to which `P[i,i]` reflects accumulated process noise.
    /// Advanced by `predict_event_to`; never moves backward.
    last_predicted_ns: Vec<u64>,
    /// Cumulative number of physical samples seen per event.
    sample_counts: Vec<u64>,
    /// Within-quantum stddev of per-sample rates from the most recent measurement per event.
    last_stddev: Vec<f64>,
}

impl ThreadFilter {
    /// Return the index for `event_id`, inserting a new state entry if it has not been seen before.
    fn ensure_event(
        &mut self,
        event_id: EventId,
        config: &KalmanConfig,
        correlation: Option<&CorrelationData>,
    ) -> usize {
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

        // Seed off-diagonal P[new, existing] = r * sqrt(P[existing,existing] * initial_variance).
        // Uses the actual current diagonal (not calibration variance) so the seed satisfies
        // |P[i,j]| <= sqrt(P[i,i] * P[j,j]) for any post-measurement P[i,i]. quantum_step
        // is the primary mechanism that maintains cross-covariance each quantum.
        let new_p_col: Vec<f64> = self
            .index_event
            .iter()
            .take(idx) // exclude the newly appended entry
            .map(|&existing_id| {
                if let (Some(c), Some(ci), Some(cj)) = (
                    correlation,
                    correlation.and_then(|c| c.idx(existing_id)),
                    correlation.and_then(|c| c.idx(event_id)),
                ) {
                    let r = c.get(ci, cj);
                    if r == 0.0 {
                        return 0.0;
                    }
                    // Seed using the actual current P diagonal for the existing event.
                    // Using calibration variances instead can violate |P[i,j]| <= sqrt(P[i,i]*P[j,j])
                    // when calibration variances exceed the post-measurement diagonal. quantum_step
                    // is the primary mechanism that maintains cross-covariance each quantum.
                    let existing_filter_idx = *self.event_index.get(&existing_id).unwrap();
                    let p_existing = self.p[existing_filter_idx][existing_filter_idx];
                    return r * (p_existing * config.initial_variance).sqrt();
                }
                0.0
            })
            .collect();

        for (k, row) in self.p.iter_mut().enumerate() {
            row.push(new_p_col[k]);
        }
        let mut new_row = vec![0.0; idx + 1];
        new_row[..idx].copy_from_slice(&new_p_col);
        new_row[idx] = config.initial_variance;
        self.p.push(new_row);
        idx
    }

    /// Inflate `P[event_idx, event_idx]` by `Q · (target_ns − last_predicted_ns[event_idx])`.
    ///
    /// Idempotent; never moves the event's clock backward. Only touches the single
    /// diagonal entry — correct under `F = I` and diagonal `Q`. Off-diagonal process
    /// noise is handled separately by `quantum_step`.
    fn predict_event_to(&mut self, event_idx: usize, target_ns: u64, config: &KalmanConfig) {
        if target_ns > self.last_predicted_ns[event_idx] {
            let delta = target_ns - self.last_predicted_ns[event_idx];
            self.p[event_idx][event_idx] += config.process_noise_per_ns * delta as f64;
            self.last_predicted_ns[event_idx] = target_ns;
        }
    }

    /// Apply a scalar Kalman measurement update for `event_idx` with observation `z` and noise `r`.
    ///
    /// Uses a one-hot `H` (only `event_idx` observed), propagating the gain through the full
    /// state vector and covariance matrix, then symmetrizes `P` to suppress floating-point drift.
    #[allow(clippy::needless_range_loop)]
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

/// Multivariate Kalman filter estimator; maintains one `ThreadFilter` per thread.
pub struct KalmanFilterEstimator {
    /// Per-thread filter state, keyed by thread id.
    threads: HashMap<u32, ThreadFilter>,
    /// Denormalized snapshot of the latest `CounterEstimate` per (tid, event_id), returned by
    /// `all_estimates` and the `rate`/`uncertainty` accessors.
    snapshots: HashMap<EstimateKey, CounterEstimate>,
    /// Filter tuning parameters.
    config: KalmanConfig,
    /// Optional calibration-derived cross-event correlations; `None` when not loaded.
    correlation: Option<CorrelationData>,
}

impl KalmanFilterEstimator {
    /// Create a new estimator with default Kalman parameters and no correlation data.
    pub fn new() -> Self {
        Self::with_config(KalmanConfig::default())
    }

    /// Build from config. Does **not** load correlation data. If
    /// `config.correlation_path` is set, call [`Self::load_correlation`] after
    /// construction (once the event registry is available) to initialize
    /// cross-event correlation. Until `load_correlation` is called the
    /// estimator uses zero off-diagonal terms.
    pub fn with_config(config: KalmanConfig) -> Self {
        Self {
            threads: HashMap::new(),
            snapshots: HashMap::new(),
            config,
            correlation: None,
        }
    }

    /// Load correlation data from `config.correlation_path`, mapping JSON event
    /// names to profiler `EventId`s via the provided name→id map.
    ///
    /// Call this once after construction, before the first profiling step, when
    /// the event registry is available.  If `correlation_path` is `None` or
    /// loading fails, the estimator continues with zero off-diagonal terms.
    pub fn load_correlation(&mut self, name_to_id: &HashMap<String, EventId>) {
        let Some(path) = &self.config.correlation_path else {
            return;
        };
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[kalman] failed to read {}: {e}", path.display());
                return;
            }
        };
        let v: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[kalman] failed to parse {}: {e}", path.display());
                return;
            }
        };
        let Some(names) = v["event_names"].as_array() else {
            eprintln!("[kalman] correlation.json missing 'event_names'");
            return;
        };
        let Some(corr_rows) = v["correlation"].as_array() else {
            eprintln!("[kalman] correlation.json missing 'correlation'");
            return;
        };
        let var_arr = v["variance"].as_array();

        // Build index: JSON position → EventId (skip unknown events).
        let json_to_event: Vec<Option<EventId>> = names
            .iter()
            .map(|n| n.as_str().and_then(|s| name_to_id.get(s)).copied())
            .collect();

        let mut correlations = HashMap::new();
        for (i, row) in corr_rows.iter().enumerate() {
            let Some(eid_i) = json_to_event.get(i).and_then(|x| *x) else {
                continue;
            };
            let Some(cols) = row.as_array() else { continue };
            for (j, val) in cols.iter().enumerate() {
                if j <= i {
                    continue;
                }
                let Some(eid_j) = json_to_event.get(j).and_then(|x| *x) else {
                    continue;
                };
                let r = val.as_f64().unwrap_or(0.0);
                if r.abs() > 1e-9 {
                    let (lo, hi) = if eid_i < eid_j {
                        (eid_i, eid_j)
                    } else {
                        (eid_j, eid_i)
                    };
                    // Store keyed by EventId pair so ThreadFilter can look up by event.
                    correlations.insert((lo as usize, hi as usize), r);
                }
            }
        }

        let mut variances_by_id: HashMap<EventId, f64> = HashMap::new();
        if let Some(va) = var_arr {
            for (i, val) in va.iter().enumerate() {
                if let Some(eid) = json_to_event.get(i).and_then(|x| *x) {
                    variances_by_id.insert(eid, val.as_f64().unwrap_or(0.0));
                }
            }
        }

        // Build a flat variances vec and event_index indexed by EventId value.
        // We want CorrelationData::idx(event_id) to return Some(event_id as usize)
        // so get/variance_for work directly with EventId.
        let max_id = name_to_id.values().copied().max().unwrap_or(0) as usize;
        let mut variances = vec![0.0f64; max_id + 1];
        let mut event_index: HashMap<EventId, usize> = HashMap::new();
        for (&eid, &var) in &variances_by_id {
            let idx = eid as usize;
            if idx < variances.len() {
                variances[idx] = var;
            }
            event_index.insert(eid, idx);
        }
        // Also register events not in variance list so idx() returns Some.
        for eid in name_to_id.values().copied() {
            event_index.entry(eid).or_insert(eid as usize);
        }

        self.correlation = Some(CorrelationData {
            correlations,
            variances,
            event_index,
        });
    }

    /// Return a reference to the active configuration.
    pub fn config(&self) -> &KalmanConfig {
        &self.config
    }

    /// Convert a covariance diagonal value to a [0, 1] uncertainty score.
    fn uncertainty_from_variance(&self, var: f64) -> f64 {
        let v = var.max(0.0);
        let denom = v + self.config.uncertainty_reference_variance;
        if denom <= 0.0 {
            1.0
        } else {
            (v / denom).clamp(0.0, 1.0)
        }
    }

    /// Rebuild the `snapshots` map for `tid` from the current filter state.
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
        let corr = self.correlation.as_ref();
        let filter = self.threads.entry(tid).or_default();
        let idx = filter.ensure_event(event_id, &self.config, corr);
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
        let corr = self.correlation.as_ref();
        let filter = self.threads.entry(tid).or_default();
        let idx = filter.ensure_event(event_id, &self.config, corr);
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

    fn quantum_step(&mut self, tid: u32, elapsed_ns: u64) {
        let Some(corr) = &self.correlation else {
            return;
        };
        let Some(filter) = self.threads.get_mut(&tid) else {
            return;
        };
        let dq_diag = self.config.process_noise_per_ns * elapsed_ns as f64;
        let scale = self.config.correlation_process_noise_scale;
        for (&(ci, cj), &r) in &corr.correlations {
            let Some(&fi) = filter.event_index.get(&(ci as EventId)) else {
                continue;
            };
            let Some(&fj) = filter.event_index.get(&(cj as EventId)) else {
                continue;
            };
            let delta = scale * dq_diag * r;
            filter.p[fi][fj] += delta;
            filter.p[fj][fi] += delta;
        }
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

    fn make_kf_with_correlation(r: f64) -> KalmanFilterEstimator {
        // Synthesize a CorrelationData directly (bypassing JSON load) to allow
        // testing the correlation path without a file on disk.
        let mut corr_map = HashMap::new();
        // EventId 0 and EventId 1; store as (0,1) since 0 < 1.
        if r.abs() > 1e-9 {
            corr_map.insert((0usize, 1usize), r);
        }
        // Variances: use a symmetric value so P seed is symmetric.
        let variance = 1e-12_f64;
        let variances = vec![variance; 2];
        let mut event_index = HashMap::new();
        event_index.insert(0u32, 0usize);
        event_index.insert(1u32, 1usize);
        let correlation = Some(CorrelationData {
            correlations: corr_map,
            variances,
            event_index,
        });
        let config = KalmanConfig::default();
        KalmanFilterEstimator {
            threads: HashMap::new(),
            snapshots: HashMap::new(),
            config,
            correlation,
        }
    }

    #[test]
    fn correlated_measurement_propagates_to_partner() {
        // With strong positive correlation r=0.8, observing event 0 at a rate
        // higher than its current estimate should shift event 1's estimate in
        // the same direction via the off-diagonal cross-covariance.
        let mut kf = make_kf_with_correlation(0.8);

        // Warm-up: establish initial estimates for both events.
        kf.measurement_update(1, 0, 1e-6, 1e-12, 100, 1_000);
        kf.measurement_update(1, 1, 1e-6, 1e-12, 100, 1_000);
        let rate_b_before = kf.rate(1, 1);

        // Now observe event 0 at a much higher rate.  Event 1 should shift up via
        // cross-covariance maintained by quantum_step (the production code path in
        // Profiler::update_estimator calls quantum_step each quantum).
        let elapsed_ns: u64 = 1_000;
        for t in 2..20 {
            kf.quantum_step(1, elapsed_ns);
            kf.measurement_update(1, 0, 3e-6, 1e-12, 100, t * elapsed_ns);
            kf.time_update(1, 1, elapsed_ns);
        }
        let rate_b_after = kf.rate(1, 1);
        // Require a shift of at least 1% of r * z_innovation as a conservative floor.
        // A test that only checks > rate_b_before passes on IEEE 754 noise; this
        // threshold requires genuine signal from the cross-covariance mechanism.
        let z_innovation = 3e-6 - 1e-6_f64;
        let min_shift = 0.01 * 0.8 * z_innovation;
        assert!(
            rate_b_after > rate_b_before + min_shift,
            "correlated partner rate should increase by at least {min_shift:.1e}: \
             before={rate_b_before:.3e} after={rate_b_after:.3e} (shift={:.3e})",
            rate_b_after - rate_b_before
        );
    }

    #[test]
    fn uncorrelated_measurement_does_not_propagate() {
        // With r=0 (no correlation), observing event 0 should not move event 1's
        // estimate (the filter reduces to independent scalar filters).
        let mut kf = make_kf_with_correlation(0.0);

        kf.measurement_update(1, 0, 1e-6, 1e-12, 100, 1_000);
        kf.measurement_update(1, 1, 1e-6, 1e-12, 100, 1_000);
        let rate_b_before = kf.rate(1, 1);

        for t in 2..20 {
            kf.measurement_update(1, 0, 3e-6, 1e-12, 100, t * 1_000);
            kf.time_update(1, 1, 1_000);
        }
        let rate_b_after = kf.rate(1, 1);
        // Without correlation, event 1's estimate should be unchanged
        // (only decaying toward 0 through lack of direct observation).
        assert_eq!(
            rate_b_after, rate_b_before,
            "independent event rate should not change from partner observations"
        );
    }

    #[test]
    fn quantum_step_adds_off_diagonal_noise() {
        let mut kf = make_kf_with_correlation(0.5);
        kf.measurement_update(1, 0, 1e-6, 1e-12, 100, 1_000);
        kf.measurement_update(1, 1, 1e-6, 1e-12, 100, 1_000);

        let p_off_before = {
            let f = &kf.threads[&1];
            let i0 = f.event_index[&0];
            let i1 = f.event_index[&1];
            f.p[i0][i1]
        };

        // quantum_step should add correlated process noise to P[0,1].
        kf.quantum_step(1, 1_000_000);

        let p_off_after = {
            let f = &kf.threads[&1];
            let i0 = f.event_index[&0];
            let i1 = f.event_index[&1];
            f.p[i0][i1]
        };

        assert!(
            p_off_after >= p_off_before,
            "off-diagonal P should not decrease after quantum_step with positive r"
        );
    }

    #[test]
    fn load_correlation_from_json_seeds_off_diagonal() {
        // Write a minimal correlation.json to a tempfile and verify that
        // load_correlation correctly parses it and seeds P off-diagonals.
        use std::io::Write;
        let json = r#"{
            "event_names": ["ev_a", "ev_b"],
            "correlation": [[1.0, 0.75], [0.75, 1.0]],
            "variance": [1e-12, 1e-12],
            "n_coobserved": [[0, 100], [100, 0]],
            "is_same_batch": [[true, true], [true, true]]
        }"#;
        let path = std::env::temp_dir().join(format!(
            "saccade_corr_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(json.as_bytes())
            .unwrap();

        let config = KalmanConfig {
            correlation_path: Some(path.clone()),
            ..KalmanConfig::default()
        };
        let mut kf = KalmanFilterEstimator::with_config(config);

        let mut name_to_id = HashMap::new();
        name_to_id.insert("ev_a".to_string(), 10u32);
        name_to_id.insert("ev_b".to_string(), 20u32);
        kf.load_correlation(&name_to_id);

        let _ = std::fs::remove_file(path);

        // Both events should be registered in CorrelationData, with r=0.75 between them.
        let corr = kf
            .correlation
            .as_ref()
            .expect("correlation should be loaded");
        let ci = corr.idx(10).expect("ev_a should have an index");
        let cj = corr.idx(20).expect("ev_b should have an index");
        let r = corr.get(ci, cj);
        assert!((r - 0.75).abs() < 1e-9, "expected r=0.75, got {r}");

        // Introduce event 10 via measurement_update, then introduce event 20 via
        // time_update (which calls ensure_event but not scalar_update). Checking P
        // right after time_update lets us see the seed before any scalar update reduces it.
        kf.measurement_update(1, 10, 1e-6, 1e-12, 100, 1_000);
        kf.time_update(1, 20, 1_000);

        let f = &kf.threads[&1];
        let fa = f.event_index[&10];
        let fb = f.event_index[&20];
        // After the fix: Seed = r * sqrt(P[10,10]_current * initial_variance).
        // Event 10 was just measured so P[10,10] ~ R = 1e-18; initial_variance = 1e-6.
        // Seed = 0.75 * sqrt(1e-18 * 1e-6) = 7.5e-13.
        assert!(
            f.p[fa][fb].abs() > 1e-20,
            "P off-diagonal should be seeded from calibration data, got {}",
            f.p[fa][fb]
        );
    }

    #[test]
    fn seed_does_not_violate_psd_when_var_cal_large() {
        // When calibration variance > sqrt(R * initial_variance), the old formula
        // r * sqrt(var_cal_i * var_cal_j) would produce a seed violating PSD.
        // The fix uses r * sqrt(P[i,i]_current * initial_variance) instead.
        let mut corr_map = HashMap::new();
        corr_map.insert((0usize, 1usize), 0.8_f64);
        let variances = vec![1e-10_f64; 2]; // large cal variance >> sqrt(R * init_var)
        let mut event_index = HashMap::new();
        event_index.insert(0u32, 0usize);
        event_index.insert(1u32, 1usize);
        let correlation = Some(CorrelationData {
            correlations: corr_map,
            variances,
            event_index,
        });
        let config = KalmanConfig::default();
        let mut kf = KalmanFilterEstimator {
            threads: HashMap::new(),
            snapshots: HashMap::new(),
            config,
            correlation,
        };

        // After event 0 is measured, P[0,0] ~ R = 1e-18.
        kf.measurement_update(1, 0, 1e-6, 1e-12, 100, 1_000);
        // Introduce event 1 via time_update (calls ensure_event).
        kf.time_update(1, 1, 1_000);

        let f = &kf.threads[&1];
        let i0 = f.event_index[&0];
        let i1 = f.event_index[&1];
        let p00 = f.p[i0][i0];
        let p11 = f.p[i1][i1];
        let p01 = f.p[i0][i1].abs();
        let psd_limit = (p00 * p11).sqrt();
        assert!(
            p01 <= psd_limit * (1.0 + 1e-10),
            "seed violates PSD: |P[0,1]|={p01:.3e} > sqrt(P[0,0]*P[1,1])={psd_limit:.3e}"
        );
    }
}
