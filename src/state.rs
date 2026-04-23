pub mod ema;
pub mod propagate;

use crate::event::EventId;
use std::collections::HashMap;

/// Per-(tid, event) snapshot produced by a `StateEstimator`.
#[derive(Debug, Clone)]
pub struct CounterEstimate {
    /// Estimated event rate (events per nanosecond).
    pub rate: f64,
    /// Within-quantum stddev of per-sample rates; 0.0 if < 2 samples.
    pub rate_stddev: f64,
    /// Uncertainty in the rate estimate — [0 = fully confident, 1 = fully uncertain].
    pub uncertainty: f64,
    /// Timestamp (ns) of the last measurement update.
    pub last_updated_ns: u64,
    /// Total number of physical samples received for this counter.
    pub sample_count: u64,
}

impl Default for CounterEstimate {
    fn default() -> Self {
        Self {
            rate: 0.0,
            rate_stddev: 0.0,
            uncertainty: 1.0,
            last_updated_ns: 0,
            sample_count: 0,
        }
    }
}

/// Key into estimator storage. Per-thread, per-event.
pub type EstimateKey = (u32, EventId);

/// Pluggable state estimator for per-(thread, counter) rate and uncertainty.
///
/// Estimators are keyed by `(tid, event_id)` and grow on demand: a key is
/// created the first time `measurement_update` is called for it. The profiler
/// then calls `time_update` on every existing key that was not observed during
/// a given quantum.
pub trait StateEstimator {
    /// Update the estimate for a (tid, event_id) pair that was observed this quantum.
    fn measurement_update(
        &mut self,
        tid: u32,
        event_id: EventId,
        rate: f64,
        stddev: f64,
        num_samples: u32,
        timestamp_ns: u64,
    );

    /// Age the estimate for a (tid, event_id) pair that was NOT observed this quantum.
    fn time_update(&mut self, tid: u32, event_id: EventId, elapsed_ns: u64);

    fn rate(&self, tid: u32, event_id: EventId) -> f64;
    fn uncertainty(&self, tid: u32, event_id: EventId) -> f64;

    /// All (tid, event_id) estimates currently tracked.
    fn all_estimates(&self) -> &HashMap<EstimateKey, CounterEstimate>;
}
