use crate::event_registry::EventId;

// Re-export from sample so existing code that imports from counter_backend still compiles.
pub use crate::sample::{MAX_COUNTERS, MAX_CPUS, TASK_COMM_LEN, WireSample};

/// Backwards-compat alias used by the sweep dummy channel and CSV logger.
/// New code should use `WireSample` directly.
pub type SaccadeSample = WireSample;

/// Aggregated observation for a single event from the previous quantum.
pub struct Observation {
    pub event_id: EventId,
    pub total_count: u64,
    pub total_duration_ns: u64,
    /// Mean event rate (events/ns) across all samples this quantum.
    pub mean_rate: f64,
    /// Population stddev of per-sample rates; 0.0 when num_samples < 2.
    pub stddev_rate: f64,
    pub min_rate: f64,
    pub max_rate: f64,
    pub num_samples: u32,
}

/// Abstraction over the source of performance counter data.
///
/// `HardwareBackend` provides real eBPF + perf counter data.
/// `VirtualBackend` generates synthetic data from golden rates.
pub trait CounterBackend {
    /// Poll for new observations and return aggregated per-event data.
    fn poll_observations(&mut self) -> Vec<Observation>;

    /// Switch active counters. Called with the old and new active sets
    /// so the backend can diff and only update changed slots.
    fn update_counters(
        &mut self,
        old_set: &[EventId],
        new_set: &[EventId],
    ) -> Result<(), Box<dyn std::error::Error>>;
}
