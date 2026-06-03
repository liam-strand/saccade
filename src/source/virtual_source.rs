//! Simulation-backed `SampleSource` that produces synthetic samples from pre-recorded rate profiles.

use std::collections::HashMap;
use std::sync::Arc;

use crate::event::EventId;
use crate::sample::RawSample;
use crate::source::{SampleSource, SwapStats};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Simulation-backed sample source.
///
/// Generates synthetic `RawSample` values from time-varying per-thread event rate profiles
/// (typically loaded from a sweep Perfetto trace). No hardware interaction.
pub struct VirtualSampleSource {
    /// Time-varying per-(event, thread) rate data used to generate samples.
    /// Wrapped in `Arc` so multiple parallel simulations can share the same read-only rates.
    rates: Arc<TimeVaryingRates>,
    /// Fractional standard deviation of Gaussian noise added to each sample's count.
    noise_stddev: f64,
    /// Events currently enabled; updated by `apply_schedule`.
    active_set: Vec<EventId>,
    /// Duration of each simulated quantum in nanoseconds.
    quantum_ns: u64,
    /// Duration of each intra-quantum sample interval in nanoseconds.
    sample_ns: u64,
    /// RNG used for Gaussian noise sampling.
    rng: StdRng,
    /// Simulated wall-clock time at the start of the current quantum.
    current_time_ns: u64,
    /// Number of counter slots exposed to the scheduler.
    num_slots: usize,
    /// Pre-computed map from event_id to sorted list of tids that have rate data for it.
    tids_by_event: HashMap<EventId, Vec<u32>>,
}

impl VirtualSampleSource {
    /// Construct a `VirtualSampleSource` from a rate profile.
    ///
    /// Pass `seed: Some(n)` for reproducible runs; `None` seeds from the OS.
    pub fn new(
        rates: Arc<TimeVaryingRates>,
        noise_stddev: f64,
        quantum_ns: u64,
        sample_ns: u64,
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
            sample_ns: sample_ns.max(1),
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
        let n = (self.quantum_ns / self.sample_ns).max(1);

        for &event_id in &self.active_set {
            let Some(tids) = self.tids_by_event.get(&event_id) else {
                continue;
            };

            for &tid in tids {
                for i in 0..n {
                    let sub_start = self.current_time_ns + i * self.sample_ns;
                    let base_rate = self.rates.rate_at(event_id, tid, sub_start);
                    let lambda = base_rate * self.sample_ns as f64;

                    let count = if lambda > 0.0 && self.noise_stddev > 0.0 {
                        let normal = Normal::new(lambda, self.noise_stddev * lambda).unwrap();
                        normal.sample(&mut self.rng).max(0.0) as u64
                    } else {
                        lambda as u64
                    };

                    samples.push(RawSample {
                        timestamp_ns: sub_start + self.sample_ns,
                        duration_ns: self.sample_ns,
                        cpu_id: 0,
                        pid: 0,
                        tid,
                        event_id,
                        count,
                        task: *b"simulate\0\0\0\0\0\0\0\0",
                    });
                }
            }
        }

        self.current_time_ns += self.quantum_ns;
        (samples, self.quantum_ns)
    }

    fn apply_schedule(
        &mut self,
        old_set: &[EventId],
        new_set: &[EventId],
    ) -> Result<SwapStats, Box<dyn std::error::Error>> {
        // Simulation does no real reconfiguration; report only how many slots
        // changed so the timing fields stay zero (and honest) for virtual runs.
        let slots_changed = if old_set.is_empty() {
            new_set.len()
        } else {
            old_set
                .iter()
                .zip(new_set.iter())
                .filter(|(old, new)| old != new)
                .count()
        };
        self.active_set = new_set.to_vec();
        Ok(SwapStats {
            slots_changed,
            ..SwapStats::default()
        })
    }

    fn num_slots(&self) -> usize {
        self.num_slots
    }
}

/// Per-(event, thread) time-varying event rates for simulation.
///
/// Keys are `(event_id, tid)`; for single-threaded data use `tid = 0`.
/// Each value is a sorted `Vec` of `(timestamp_ns, rate_events_per_ns)` breakpoints.
pub struct TimeVaryingRates {
    /// Sorted breakpoint series keyed by (event_id, tid).
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
