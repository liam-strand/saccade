use crate::counter_backend::{CounterBackend, MAX_COUNTERS, Observation, SaccadeSample};
use crate::event_registry::EventId;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::sync::mpsc::SyncSender;

/// Per-event time-varying rates, keyed by EventId.
/// Each entry is a sorted Vec of (timestamp_ns, rate_events_per_ns).
pub struct TimeVaryingRates {
    pub series: HashMap<EventId, Vec<(u64, f64)>>,
}

impl TimeVaryingRates {
    /// Return the interpolated rate for `event_id` at `time_ns`.
    /// Holds the first/last observed rate before/after the recorded range.
    pub fn rate_at(&self, event_id: EventId, time_ns: u64) -> f64 {
        let Some(points) = self.series.get(&event_id) else {
            return 0.0;
        };
        if points.is_empty() {
            return 0.0;
        }
        if time_ns <= points[0].0 {
            return points[0].1;
        }
        let last = points[points.len() - 1];
        if time_ns >= last.0 {
            return last.1;
        }
        // Find the two surrounding points via binary search.
        let idx = points.partition_point(|&(ts, _)| ts <= time_ns);
        let (t0, r0) = points[idx - 1];
        let (t1, r1) = points[idx];
        let frac = (time_ns - t0) as f64 / (t1 - t0) as f64;
        r0 + frac * (r1 - r0)
    }
}

pub struct VirtualBackend {
    rates: TimeVaryingRates,
    noise_stddev: f64,
    active_set: Vec<EventId>,
    quantum_ns: u64,
    logger_tx: Option<SyncSender<SaccadeSample>>,
    rng: StdRng,
    current_time_ns: u64,
}

impl VirtualBackend {
    pub fn new(
        rates: TimeVaryingRates,
        noise_stddev: f64,
        quantum_ns: u64,
        seed: Option<u64>,
        logger_tx: Option<SyncSender<SaccadeSample>>,
    ) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        Self {
            rates,
            noise_stddev,
            active_set: Vec::new(),
            quantum_ns,
            logger_tx,
            rng,
            current_time_ns: 0,
        }
    }
}

impl CounterBackend for VirtualBackend {
    fn poll_observations(&mut self) -> Vec<Observation> {
        let mut observations = Vec::new();

        let mut sample = SaccadeSample {
            timestamp_ns: self.current_time_ns,
            duration_ns: self.quantum_ns,
            task: *b"simulate\0\0\0\0\0\0\0\0",
            ..Default::default()
        };

        for (slot, &event_id) in self.active_set.iter().enumerate() {
            let base_rate = self.rates.rate_at(event_id, self.current_time_ns);
            let lambda = base_rate * self.quantum_ns as f64;

            let count = if lambda > 0.0 && self.noise_stddev > 0.0 {
                let normal = Normal::new(lambda, self.noise_stddev * lambda).unwrap();
                normal.sample(&mut self.rng).max(0.0) as u64
            } else {
                lambda as u64
            };

            let mean_rate = count as f64 / self.quantum_ns as f64;
            observations.push(Observation {
                event_id,
                total_count: count,
                total_duration_ns: self.quantum_ns,
                mean_rate,
                stddev_rate: 0.0,
                min_rate: mean_rate,
                max_rate: mean_rate,
                num_samples: 1,
            });

            if slot < MAX_COUNTERS {
                sample.counters[slot] = count;
                sample.events[slot] = event_id as u64;
            }
        }

        if let Some(tx) = &self.logger_tx {
            let _ = tx.try_send(sample);
        }

        self.current_time_ns += self.quantum_ns;
        observations
    }

    fn update_counters(
        &mut self,
        _old_set: &[EventId],
        new_set: &[EventId],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.active_set = new_set.to_vec();
        Ok(())
    }
}
