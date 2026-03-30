use crate::counter_backend::{CounterBackend, MAX_COUNTERS, Observation, SaccadeSample};
use crate::event_registry::EventId;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::mpsc::SyncSender;

#[derive(Deserialize)]
pub struct GoldenRates {
    pub rates: HashMap<String, f64>,
    #[serde(default)]
    pub noise_stddev: f64,
    pub seed: Option<u64>,
}

pub struct VirtualBackend {
    golden_rates: HashMap<EventId, f64>,
    noise_stddev: f64,
    active_set: Vec<EventId>,
    quantum_ns: u64,
    logger_tx: Option<SyncSender<SaccadeSample>>,
    rng: StdRng,
    current_time_ns: u64,
}

impl VirtualBackend {
    pub fn new(
        golden_rates: HashMap<EventId, f64>,
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
            golden_rates,
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
            let base_rate = self.golden_rates.get(&event_id).copied().unwrap_or(0.0);
            let lambda = base_rate * self.quantum_ns as f64;

            let count = if lambda > 0.0 && self.noise_stddev > 0.0 {
                let normal = Normal::new(lambda, self.noise_stddev * lambda).unwrap();
                normal.sample(&mut self.rng).max(0.0) as u64
            } else {
                lambda as u64
            };

            observations.push(Observation {
                event_id,
                total_count: count,
                total_duration_ns: self.quantum_ns,
            });

            if slot < MAX_COUNTERS {
                sample.values[slot] = count;
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
