//! Exponential-moving-average state estimator.
//!
//! Each (tid, event_id) pair maintains a rate EMA and a linearly-growing
//! uncertainty score. Uncertainty resets to 0 on each observed quantum and
//! grows at `uncertainty_growth_rate` events/ns² during unobserved quanta.

use crate::event::EventId;
use crate::state::{CounterEstimate, EstimateKey, StateEstimator};
use std::collections::HashMap;

/// Configuration for the EMA-based state estimator.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmaConfig {
    /// Smoothing factor applied to new rate observations; higher values track
    /// changes faster at the cost of more variance.
    pub alpha: f64,
    /// Rate at which uncertainty grows per nanosecond of no observation
    /// (units: uncertainty/ns, in [0, 1]).
    pub uncertainty_growth_rate: f64,
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            uncertainty_growth_rate: 1e-8,
        }
    }
}

/// EMA-based state estimator; implements [`StateEstimator`] for all tracked (tid, event) pairs.
pub struct VirtualCounterState {
    /// Per-(tid, event_id) snapshot storage; grows on first observation.
    estimates: HashMap<EstimateKey, CounterEstimate>,
    /// EMA smoothing factor (mirrors `EmaConfig::alpha`).
    alpha: f64,
    /// Uncertainty growth rate per nanosecond (mirrors `EmaConfig::uncertainty_growth_rate`).
    uncertainty_growth_rate: f64,
}

impl VirtualCounterState {
    /// Create a new estimator with default EMA parameters.
    pub fn new() -> Self {
        Self::with_config(EmaConfig::default())
    }

    /// Create a new estimator with the supplied configuration.
    pub fn with_config(config: EmaConfig) -> Self {
        Self {
            estimates: HashMap::new(),
            alpha: config.alpha,
            uncertainty_growth_rate: config.uncertainty_growth_rate,
        }
    }
}

impl Default for VirtualCounterState {
    fn default() -> Self {
        Self::new()
    }
}

impl StateEstimator for VirtualCounterState {
    fn measurement_update(
        &mut self,
        tid: u32,
        event_id: EventId,
        rate: f64,
        stddev: f64,
        num_samples: u32,
        timestamp_ns: u64,
    ) {
        let est = self.estimates.entry((tid, event_id)).or_default();
        if est.sample_count == 0 {
            est.rate = rate;
        } else {
            est.rate = self.alpha * rate + (1.0 - self.alpha) * est.rate;
        }
        est.rate_stddev = stddev;
        est.uncertainty = 0.0;
        est.last_updated_ns = timestamp_ns;
        est.sample_count += num_samples as u64;
    }

    fn time_update(&mut self, tid: u32, event_id: EventId, elapsed_ns: u64) {
        if let Some(est) = self.estimates.get_mut(&(tid, event_id)) {
            est.uncertainty += self.uncertainty_growth_rate * elapsed_ns as f64;
            if est.uncertainty > 1.0 {
                est.uncertainty = 1.0;
            }
        }
    }

    fn rate(&self, tid: u32, event_id: EventId) -> f64 {
        self.estimates.get(&(tid, event_id)).map_or(0.0, |e| e.rate)
    }

    fn uncertainty(&self, tid: u32, event_id: EventId) -> f64 {
        self.estimates
            .get(&(tid, event_id))
            .map_or(1.0, |e| e.uncertainty)
    }

    fn all_estimates(&self) -> &HashMap<EstimateKey, CounterEstimate> {
        &self.estimates
    }
}
