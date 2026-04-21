use crate::event::EventId;
use crate::state::{CounterEstimate, StateEstimator};

/// Last-observation-carried-forward estimator.
///
/// On `measurement_update`, replaces the stored rate with the raw measurement.
/// On `time_update`, does nothing — uncertainty stays at 0 indefinitely.
/// Useful as a baseline: scheduler sees exactly the most recent measurement,
/// with no smoothing and no staleness penalty.
pub struct PropagateEstimator {
    estimates: Vec<CounterEstimate>,
}

impl PropagateEstimator {
    pub fn new() -> Self {
        Self {
            estimates: Vec::new(),
        }
    }
}

impl Default for PropagateEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl StateEstimator for PropagateEstimator {
    fn init(&mut self, num_events: usize) {
        self.estimates = vec![CounterEstimate::default(); num_events];
    }

    fn measurement_update(
        &mut self,
        event_id: EventId,
        rate: f64,
        stddev: f64,
        num_samples: u32,
        timestamp_ns: u64,
    ) {
        if let Some(est) = self.estimates.get_mut(event_id as usize) {
            est.rate = rate;
            est.rate_stddev = stddev;
            est.uncertainty = 0.0;
            est.last_updated_ns = timestamp_ns;
            est.sample_count += num_samples as u64;
        }
    }

    fn time_update(&mut self, _event_id: EventId, _elapsed_ns: u64) {}

    fn num_events(&self) -> usize {
        self.estimates.len()
    }

    fn rate(&self, event_id: EventId) -> f64 {
        self.estimates
            .get(event_id as usize)
            .map_or(0.0, |e| e.rate)
    }

    fn rate_stddev(&self, event_id: EventId) -> f64 {
        self.estimates
            .get(event_id as usize)
            .map_or(0.0, |e| e.rate_stddev)
    }

    fn uncertainty(&self, event_id: EventId) -> f64 {
        self.estimates
            .get(event_id as usize)
            .map_or(1.0, |e| e.uncertainty)
    }

    fn sample_count(&self, event_id: EventId) -> u64 {
        self.estimates
            .get(event_id as usize)
            .map_or(0, |e| e.sample_count)
    }

    fn all_estimates(&self) -> &[CounterEstimate] {
        &self.estimates
    }
}
