use crate::event::EventId;
use crate::state::{CounterEstimate, EstimateKey, StateEstimator};
use std::collections::HashMap;

/// Last-observation-carried-forward estimator.
///
/// On `measurement_update`, replaces the stored rate with the raw measurement.
/// On `time_update`, does nothing — uncertainty stays at 0 indefinitely.
pub struct PropagateEstimator {
    estimates: HashMap<EstimateKey, CounterEstimate>,
}

impl PropagateEstimator {
    pub fn new() -> Self {
        Self {
            estimates: HashMap::new(),
        }
    }
}

impl Default for PropagateEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl StateEstimator for PropagateEstimator {
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
        est.rate = rate;
        est.rate_stddev = stddev;
        est.uncertainty = 0.0;
        est.last_updated_ns = timestamp_ns;
        est.sample_count += num_samples as u64;
    }

    fn time_update(&mut self, _tid: u32, _event_id: EventId, _elapsed_ns: u64) {}

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
