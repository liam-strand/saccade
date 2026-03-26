use crate::event_registry::EventId;

/// Per-counter estimate, maintained for every event in the registry.
/// Tracks a running rate estimate and an uncertainty measure.
#[derive(Debug, Clone)]
pub struct CounterEstimate {
    /// Exponential moving average of event rate (events per nanosecond)
    pub rate: f64,
    /// Uncertainty in the rate estimate — grows when the counter is inactive
    pub uncertainty: f64,
    /// Timestamp (ns) of the last measurement update
    pub last_updated_ns: u64,
    /// Total number of physical samples received for this counter
    pub sample_count: u64,
}

impl Default for CounterEstimate {
    fn default() -> Self {
        Self {
            rate: 0.0,
            uncertainty: 1.0, // start fully uncertain
            last_updated_ns: 0,
            sample_count: 0,
        }
    }
}

/// Tracks rate estimates and uncertainty for all counters in the event universe.
///
/// Oculomotor owns this struct and updates it each quantum. The scheduler
/// receives a `&VirtualCounterState` to inform its next scheduling decision.
pub struct VirtualCounterState {
    estimates: Vec<CounterEstimate>,
    /// EMA smoothing factor (0..1). Higher = more weight on recent observations.
    alpha: f64,
    /// Rate at which uncertainty grows per nanosecond of inactivity.
    uncertainty_growth_rate: f64,
}

impl VirtualCounterState {
    /// Create state for `num_events` counters with default EMA parameters.
    pub fn new(num_events: usize) -> Self {
        Self {
            estimates: vec![CounterEstimate::default(); num_events],
            alpha: 0.3,
            uncertainty_growth_rate: 1e-6, // per nanosecond
        }
    }

    /// Update the rate estimate for a counter that was physically sampled this quantum.
    ///
    /// `rate` is the measured event rate (events/ns) aggregated across all CPUs.
    /// `timestamp_ns` is the current time.
    pub fn measurement_update(&mut self, event_id: EventId, rate: f64, timestamp_ns: u64) {
        if let Some(est) = self.estimates.get_mut(event_id as usize) {
            if est.sample_count == 0 {
                // First observation: initialize directly
                est.rate = rate;
            } else {
                // EMA update
                est.rate = self.alpha * rate + (1.0 - self.alpha) * est.rate;
            }
            est.uncertainty = 0.0; // just observed — full confidence
            est.last_updated_ns = timestamp_ns;
            est.sample_count += 1;
        }
    }

    /// Grow uncertainty for a counter that was NOT sampled this quantum.
    ///
    /// `elapsed_ns` is the time since the last step (approximately one quantum).
    pub fn time_update(&mut self, event_id: EventId, elapsed_ns: u64) {
        if let Some(est) = self.estimates.get_mut(event_id as usize) {
            est.uncertainty += self.uncertainty_growth_rate * elapsed_ns as f64;
            // Clamp to [0, 1]
            if est.uncertainty > 1.0 {
                est.uncertainty = 1.0;
            }
        }
    }

    pub fn rate(&self, event_id: EventId) -> f64 {
        self.estimates
            .get(event_id as usize)
            .map_or(0.0, |e| e.rate)
    }

    pub fn uncertainty(&self, event_id: EventId) -> f64 {
        self.estimates
            .get(event_id as usize)
            .map_or(1.0, |e| e.uncertainty)
    }

    pub fn num_events(&self) -> usize {
        self.estimates.len()
    }

    pub fn all_estimates(&self) -> &[CounterEstimate] {
        &self.estimates
    }
}
