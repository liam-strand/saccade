pub mod ema;

use crate::event::EventId;

/// Per-counter snapshot produced by a `StateEstimator`.
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

/// Pluggable state estimator for per-counter rate and uncertainty.
///
/// The profiler calls `init` once at build time, then `measurement_update` /
/// `time_update` each quantum. Schedulers and sinks query state via the read
/// accessors and the optional `correlation` hook.
pub trait StateEstimator {
    /// Size internal storage for `num_events` counters.
    fn init(&mut self, num_events: usize);

    /// Update the estimate for a counter that was observed this quantum.
    fn measurement_update(
        &mut self,
        event_id: EventId,
        rate: f64,
        stddev: f64,
        num_samples: u32,
        timestamp_ns: u64,
    );

    /// Grow uncertainty for a counter that was not sampled this quantum.
    fn time_update(&mut self, event_id: EventId, elapsed_ns: u64);

    fn num_events(&self) -> usize;
    fn rate(&self, event_id: EventId) -> f64;
    fn rate_stddev(&self, event_id: EventId) -> f64;
    fn uncertainty(&self, event_id: EventId) -> f64;
    fn sample_count(&self, event_id: EventId) -> u64;

    /// Snapshot of all per-counter estimates. Implementations that do not
    /// natively store `CounterEstimate` values should maintain a materialized
    /// cache updated in `measurement_update` / `time_update`.
    fn all_estimates(&self) -> &[CounterEstimate];

    /// Pearson correlation between two counters' rates, if tracked.
    /// Returns `None` for estimators that do not model cross-counter relationships.
    fn correlation(&self, _a: EventId, _b: EventId) -> Option<f64> {
        None
    }
}
