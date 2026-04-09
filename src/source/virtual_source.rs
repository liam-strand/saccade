use crate::event_registry::EventId;
use crate::sample::RawSample;
use crate::source::SampleSource;
use crate::virtual_backend::TimeVaryingRates;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Simulation-backed sample source.
///
/// Generates synthetic `RawSample` values from time-varying event rate profiles
/// (typically loaded from a sweep Perfetto trace). No hardware interaction.
///
/// Unlike the old `VirtualBackend`, this does not construct fake `WireSample`
/// structs — it produces `RawSample` directly.
pub struct VirtualSampleSource {
    rates: TimeVaryingRates,
    noise_stddev: f64,
    active_set: Vec<EventId>,
    quantum_ns: u64,
    rng: StdRng,
    current_time_ns: u64,
    num_slots: usize,
}

impl VirtualSampleSource {
    pub fn new(
        rates: TimeVaryingRates,
        noise_stddev: f64,
        quantum_ns: u64,
        seed: Option<u64>,
        num_slots: usize,
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
            rng,
            current_time_ns: 0,
            num_slots,
        }
    }
}

impl SampleSource for VirtualSampleSource {
    fn collect(&mut self) -> (Vec<RawSample>, u64) {
        let mut samples = Vec::new();
        let ts = self.current_time_ns + self.quantum_ns;

        for &event_id in &self.active_set {
            let base_rate = self.rates.rate_at(event_id, self.current_time_ns);
            let lambda = base_rate * self.quantum_ns as f64;

            let count = if lambda > 0.0 && self.noise_stddev > 0.0 {
                let normal = Normal::new(lambda, self.noise_stddev * lambda).unwrap();
                normal.sample(&mut self.rng).max(0.0) as u64
            } else {
                lambda as u64
            };

            samples.push(RawSample {
                timestamp_ns: ts,
                duration_ns: self.quantum_ns,
                cpu_id: 0,
                pid: 0,
                event_id,
                count,
                task: *b"simulate\0\0\0\0\0\0\0\0",
            });
        }

        self.current_time_ns = ts;
        (samples, self.quantum_ns)
    }

    fn apply_schedule(
        &mut self,
        _old_set: &[EventId],
        new_set: &[EventId],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.active_set = new_set.to_vec();
        Ok(())
    }

    fn num_slots(&self) -> usize {
        self.num_slots
    }
}
