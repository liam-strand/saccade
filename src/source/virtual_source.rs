use std::collections::HashMap;

use crate::event::EventId;
use crate::sample::RawSample;
use crate::source::SampleSource;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Simulation-backed sample source.
///
/// Generates synthetic `RawSample` values from time-varying per-thread event rate profiles
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
    /// Pre-computed: event_id → sorted list of tids that have data for it.
    tids_by_event: HashMap<EventId, Vec<u32>>,
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
        // Pre-compute tids_by_event for efficient lookup.
        let mut tids_by_event: HashMap<EventId, Vec<u32>> = HashMap::new();
        for &(event_id, tid) in rates.series.keys() {
            tids_by_event.entry(event_id).or_default().push(tid);
        }
        for tids in tids_by_event.values_mut() {
            tids.sort_unstable();
        }
        Self {
            rates,
            noise_stddev,
            active_set: Vec::new(),
            quantum_ns,
            rng,
            current_time_ns: 0,
            num_slots,
            tids_by_event,
        }
    }
}

impl SampleSource for VirtualSampleSource {
    fn collect(&mut self) -> (Vec<RawSample>, u64) {
        let mut samples = Vec::new();
        let ts = self.current_time_ns + self.quantum_ns;

        for &event_id in &self.active_set {
            let tids = match self.tids_by_event.get(&event_id) {
                Some(t) => t.clone(),
                None => continue,
            };

            for tid in tids {
                let base_rate = self.rates.rate_at(event_id, tid, self.current_time_ns);
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
                    tid,
                    event_id,
                    count,
                    task: *b"simulate\0\0\0\0\0\0\0\0",
                });
            }
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

/// Per-(event, thread) time-varying rates.
/// Key: (event_id, tid). For single-threaded data, tid = 0.
/// Each entry is a sorted Vec of (timestamp_ns, rate_events_per_ns).
pub struct TimeVaryingRates {
    pub series: HashMap<(EventId, u32), Vec<(u64, f64)>>,
}

impl TimeVaryingRates {
    /// Return the interpolated rate for `(event_id, tid)` at `time_ns`.
    /// Holds the first/last observed rate before/after the recorded range.
    pub fn rate_at(&self, event_id: EventId, tid: u32, time_ns: u64) -> f64 {
        let Some(points) = self.series.get(&(event_id, tid)) else {
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
