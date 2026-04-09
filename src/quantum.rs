use crate::event::EventId;
use crate::sample::RawSample;
use std::cell::OnceCell;
use std::collections::HashMap;

/// Per-event aggregate computed from the raw samples in a `Quantum`.
/// Rate = `total_count / total_duration_ns` — computed lazily from raw counts.
#[derive(Debug, Clone)]
pub struct EventAggregate {
    pub event_id: EventId,
    pub total_count: u64,
    pub total_duration_ns: u64,
    /// Welford mean of per-sample (count / duration_ns) rates.
    pub mean_rate: f64,
    /// Welford population stddev of per-sample rates; 0.0 when num_samples < 2.
    pub stddev_rate: f64,
    pub min_rate: f64,
    pub max_rate: f64,
    pub num_samples: u32,
}

/// All raw samples collected during one scheduling step.
///
/// Raw counts and durations are the primary data; rate aggregates are computed
/// lazily on first access and cached. Consumers can choose their level of detail:
/// - CSV output: iterate `samples()` for per-sample counts + durations
/// - VCS update: use `aggregates()` for per-event mean rate + stddev
/// - Schedulers: access both
pub struct Quantum {
    samples: Vec<RawSample>,
    timestamp_ns: u64,
    elapsed_ns: u64,
    aggregates: OnceCell<HashMap<EventId, EventAggregate>>,
}

impl Quantum {
    pub fn new(samples: Vec<RawSample>, timestamp_ns: u64, elapsed_ns: u64) -> Self {
        Self {
            samples,
            timestamp_ns,
            elapsed_ns,
            aggregates: OnceCell::new(),
        }
    }

    pub fn samples(&self) -> &[RawSample] {
        &self.samples
    }

    pub fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.elapsed_ns
    }

    /// Lazily compute per-event rate aggregates using Welford's online algorithm.
    /// Rate for each sample = `count / duration_ns`. Result is cached.
    pub fn aggregates(&self) -> &HashMap<EventId, EventAggregate> {
        self.aggregates.get_or_init(|| aggregate_samples(&self.samples))
    }

    /// Returns the set of event IDs that have at least one sample in this quantum.
    pub fn observed_events(&self) -> Vec<EventId> {
        self.aggregates().keys().copied().collect()
    }
}

/// Welford's online algorithm over per-sample rates (count / duration_ns).
fn aggregate_samples(samples: &[RawSample]) -> HashMap<EventId, EventAggregate> {
    struct Acc {
        n: u32,
        mean: f64,
        m2: f64,
        min: f64,
        max: f64,
        total_count: u64,
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
