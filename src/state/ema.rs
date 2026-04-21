use crate::event::EventId;
use crate::state::{CounterEstimate, StateEstimator};

/// EMA-based state estimator.
///
/// Tracks a per-counter rate estimate (exponential moving average) and a
/// scalar uncertainty that grows linearly during inactivity.
pub struct VirtualCounterState {
    estimates: Vec<CounterEstimate>,
    /// EMA smoothing factor (0..1). Higher = more weight on recent observations.
    alpha: f64,
    /// Rate at which uncertainty grows per nanosecond of inactivity.
    uncertainty_growth_rate: f64,
}

impl VirtualCounterState {
    pub fn new(num_events: usize) -> Self {
        Self {
            estimates: vec![CounterEstimate::default(); num_events],
            alpha: 0.3,
            uncertainty_growth_rate: 1e-8,
        }
    }
}

impl StateEstimator for VirtualCounterState {
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
    }

    fn time_update(&mut self, event_id: EventId, elapsed_ns: u64) {
        if let Some(est) = self.estimates.get_mut(event_id as usize) {
            est.uncertainty += self.uncertainty_growth_rate * elapsed_ns as f64;
            if est.uncertainty > 1.0 {
                est.uncertainty = 1.0;
            }
        }
    }

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
