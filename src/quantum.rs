use crate::event::EventId;
use crate::sample::RawSample;
use std::cell::OnceCell;
use std::collections::HashMap;

/// Per-event statistics aggregated from all `RawSample`s in a `Quantum`.
#[derive(Debug, Clone)]
pub struct EventAggregate {
    /// The hardware event these statistics describe.
    pub event_id: EventId,
    /// Sum of raw event counts across all samples.
    pub total_count: u64,
    /// Sum of measurement durations across all samples in nanoseconds.
    pub total_duration_ns: u64,
    /// Welford online mean of per-sample rates (`count / duration_ns`).
    pub mean_rate: f64,
    /// Welford population standard deviation of per-sample rates; `0.0` when `num_samples < 2`.
    pub stddev_rate: f64,
    /// Minimum per-sample rate observed.
    pub min_rate: f64,
    /// Maximum per-sample rate observed.
    pub max_rate: f64,
    /// Number of raw samples that contributed to this aggregate.
    pub num_samples: u32,
}

/// All raw samples collected during one scheduling step, with lazily computed rate aggregates.
pub struct Quantum {
    /// Individual hardware counter observations collected during this scheduling step.
    samples: Vec<RawSample>,
    /// Kernel monotonic timestamp marking the end of this quantum (nanoseconds).
    timestamp_ns: u64,
    /// Wall-clock duration of this scheduling step in nanoseconds.
    elapsed_ns: u64,
    /// Per-event aggregates across all CPUs, computed on first access and cached.
    aggregates: OnceCell<HashMap<EventId, EventAggregate>>,
    /// Per-(tid, event) aggregates, computed on first access and cached.
    per_thread_aggregates: OnceCell<HashMap<(u32, EventId), EventAggregate>>,
}

impl Quantum {
    /// Construct a `Quantum` from a batch of raw samples and the scheduling-step timing metadata.
    pub fn new(samples: Vec<RawSample>, timestamp_ns: u64, elapsed_ns: u64) -> Self {
        Self {
            samples,
            timestamp_ns,
            elapsed_ns,
            aggregates: OnceCell::new(),
            per_thread_aggregates: OnceCell::new(),
        }
    }

    /// Returns the individual raw samples collected during this quantum.
    pub fn samples(&self) -> &[RawSample] {
        &self.samples
    }

    /// Returns the kernel monotonic timestamp at the end of this quantum (nanoseconds).
    pub fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    /// Returns the wall-clock duration of this scheduling step in nanoseconds.
    pub fn elapsed_ns(&self) -> u64 {
        self.elapsed_ns
    }

    /// Lazily compute per-event rate aggregates using Welford's online algorithm.
    /// Rate for each sample = `count / duration_ns`. Result is cached.
    pub fn aggregates(&self) -> &HashMap<EventId, EventAggregate> {
        self.aggregates
            .get_or_init(|| aggregate_samples(&self.samples))
    }

    /// Returns the set of event IDs that have at least one sample in this quantum.
    pub fn observed_events(&self) -> Vec<EventId> {
        self.aggregates().keys().copied().collect()
    }

    /// Lazily compute per-(tid, event) rate aggregates. Result is cached.
    pub fn per_thread_aggregates(&self) -> &HashMap<(u32, EventId), EventAggregate> {
        self.per_thread_aggregates
            .get_or_init(|| aggregate_samples_by_thread(&self.samples))
    }
}

/// Applies Welford's online algorithm over per-sample rates keyed by `(tid, event_id)`.
fn aggregate_samples_by_thread(samples: &[RawSample]) -> HashMap<(u32, EventId), EventAggregate> {
    /// Running accumulator for Welford's online mean/variance algorithm.
    struct Acc {
        /// Number of samples seen so far.
        n: u32,
        /// Running Welford mean of per-sample rates.
        mean: f64,
        /// Running Welford M2 sum of squared deviations (used to derive variance).
        m2: f64,
        /// Minimum per-sample rate seen so far.
        min: f64,
        /// Maximum per-sample rate seen so far.
        max: f64,
        /// Cumulative raw event count.
        total_count: u64,
        /// Cumulative measurement duration in nanoseconds.
        total_duration_ns: u64,
    }

    let mut by_thread_event: HashMap<(u32, EventId), Acc> = HashMap::new();

    for s in samples {
        if s.duration_ns == 0 {
            continue;
        }
        let rate = s.count as f64 / s.duration_ns as f64;
        let acc = by_thread_event.entry((s.tid, s.event_id)).or_insert(Acc {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            total_count: 0,
            total_duration_ns: 0,
        });
        acc.n += 1;
        let delta = rate - acc.mean;
        acc.mean += delta / acc.n as f64;
        acc.m2 += delta * (rate - acc.mean);
        if rate < acc.min {
            acc.min = rate;
        }
        if rate > acc.max {
            acc.max = rate;
        }
        acc.total_count += s.count;
        acc.total_duration_ns += s.duration_ns;
    }

    by_thread_event
        .into_iter()
        .map(|((tid, event_id), acc)| {
            (
                (tid, event_id),
                EventAggregate {
                    event_id,
                    total_count: acc.total_count,
                    total_duration_ns: acc.total_duration_ns,
                    mean_rate: acc.mean,
                    stddev_rate: if acc.n < 2 {
                        0.0
                    } else {
                        (acc.m2 / acc.n as f64).sqrt()
                    },
                    min_rate: if acc.n == 0 { 0.0 } else { acc.min },
                    max_rate: if acc.n == 0 { 0.0 } else { acc.max },
                    num_samples: acc.n,
                },
            )
        })
        .collect()
}

/// Applies Welford's online algorithm over per-sample rates keyed by `event_id`.
fn aggregate_samples(samples: &[RawSample]) -> HashMap<EventId, EventAggregate> {
    /// Running accumulator for Welford's online mean/variance algorithm.
    struct Acc {
        /// Number of samples seen so far.
        n: u32,
        /// Running Welford mean of per-sample rates.
        mean: f64,
        /// Running Welford M2 sum of squared deviations (used to derive variance).
        m2: f64,
        /// Minimum per-sample rate seen so far.
        min: f64,
        /// Maximum per-sample rate seen so far.
        max: f64,
        /// Cumulative raw event count.
        total_count: u64,
        /// Cumulative measurement duration in nanoseconds.
        total_duration_ns: u64,
    }

    let mut by_event: HashMap<EventId, Acc> = HashMap::new();

    for s in samples {
        if s.duration_ns == 0 {
            continue;
        }
        let rate = s.count as f64 / s.duration_ns as f64;
        let acc = by_event.entry(s.event_id).or_insert(Acc {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            total_count: 0,
            total_duration_ns: 0,
        });
        acc.n += 1;
        let delta = rate - acc.mean;
        acc.mean += delta / acc.n as f64;
        acc.m2 += delta * (rate - acc.mean);
        if rate < acc.min {
            acc.min = rate;
        }
        if rate > acc.max {
            acc.max = rate;
        }
        acc.total_count += s.count;
        acc.total_duration_ns += s.duration_ns;
    }

    by_event
        .into_iter()
        .map(|(event_id, acc)| {
            (
                event_id,
                EventAggregate {
                    event_id,
                    total_count: acc.total_count,
                    total_duration_ns: acc.total_duration_ns,
                    mean_rate: acc.mean,
                    stddev_rate: if acc.n < 2 {
                        0.0
                    } else {
                        (acc.m2 / acc.n as f64).sqrt()
                    },
                    min_rate: if acc.n == 0 { 0.0 } else { acc.min },
                    max_rate: if acc.n == 0 { 0.0 } else { acc.max },
                    num_samples: acc.n,
                },
            )
        })
        .collect()
}
