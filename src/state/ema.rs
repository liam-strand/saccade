use crate::event::EventId;
use crate::state::{CounterEstimate, EstimateKey, StateEstimator};
use std::collections::HashMap;

/// EMA-based state estimator.
///
/// Tracks a per-(tid, event) rate estimate (exponential moving average) and a
/// scalar uncertainty that grows linearly during inactivity.
pub struct VirtualCounterState {
    estimates: HashMap<EstimateKey, CounterEstimate>,
    /// EMA smoothing factor (0..1). Higher = more weight on recent observations.
    alpha: f64,
    /// Rate at which uncertainty grows per nanosecond of inactivity.
    uncertainty_growth_rate: f64,
}

impl VirtualCounterState {
    pub fn new() -> Self {
        Self {
            estimates: HashMap::new(),
            alpha: 0.3,
            uncertainty_growth_rate: 1e-8,
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
