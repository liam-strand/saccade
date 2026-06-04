//! Main profiling orchestrator: drives the collect → estimate → schedule → emit loop.

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::sample::MAX_COUNTERS;
use crate::scheduler::Scheduler;
use crate::sink::OutputSink;
use crate::source::SampleSource;
use crate::state::StateEstimator;
use std::collections::HashSet;
use std::time::Duration;

/// Main profiling orchestrator.
///
/// Each `step()`:
/// 1. Collects raw samples from the source
/// 2. Builds a `Quantum` (raw samples + lazy aggregates)
/// 3. Updates the state estimator from the quantum's aggregates
/// 4. Asks the scheduler for the next active set
/// 5. Applies the schedule to the source
/// 6. Emits the quantum + estimator state to all output sinks
pub struct Profiler<'s> {
    /// Active sample source (hardware eBPF or virtual simulation).
    source: Box<dyn SampleSource>,
    /// Policy that decides which events to monitor next and for how long.
    scheduler: Box<dyn Scheduler>,
    /// Output sinks that receive each completed quantum.
    sinks: &'s mut [Box<dyn OutputSink>],
    /// Maintains per-(thread, event) rate estimates across quanta.
    estimator: Box<dyn StateEstimator>,
    /// Events currently enabled on the hardware counters.
    active_set: Vec<EventId>,
    /// Monotonically increasing wall-clock position in nanoseconds since profiling started.
    current_time_ns: u64,
}

impl<'s> Profiler<'s> {
    /// Run one profiling quantum: collect, estimate, schedule, and emit.
    ///
    /// Returns the scheduler's requested duration for the next quantum, or `None`
    /// if the scheduler signals that profiling should stop.
    pub fn step(&mut self) -> Option<Duration> {
        // 1. Collect raw samples
        let (raw_samples, elapsed_ns) = self.source.collect();

        // 2. Build Quantum
        self.current_time_ns += elapsed_ns;
        let quantum = Quantum::new(raw_samples, self.current_time_ns, elapsed_ns);

        // 3. Update estimator
        self.update_estimator(&quantum, elapsed_ns);

        // 4. Scheduler decision
        let decision = self.scheduler.next_step(&quantum, self.estimator.as_ref());

        // 5. Apply schedule
        let swap_start = std::time::Instant::now();
        let stats = self
            .source
            .apply_schedule(&self.active_set, &decision.active_events)
            .unwrap();
        let swap_ns = swap_start.elapsed().as_nanos();
        tracing::debug!(
            swap_ns,
            quiesce_ns = stats.quiesce_ns,
            reconfig_ns = stats.reconfig_ns,
            slots_changed = stats.slots_changed,
            quantum_ns = elapsed_ns,
            "slot_swap"
        );
        self.active_set = decision.active_events;

        // 6. Emit to all sinks
        for sink in self.sinks.iter_mut() {
            let _ = sink.emit(&quantum, self.estimator.as_ref(), &self.active_set);
        }

        decision.duration
    }

    /// Update state estimates using per-thread aggregates from the just-completed quantum.
    ///
    /// Issues measurement updates for observed (tid, event) pairs and time updates for
    /// any pair that was not observed this quantum, then applies cross-event process noise.
    fn update_estimator(&mut self, quantum: &Quantum, elapsed_ns: u64) {
        let per_thread = quantum.per_thread_aggregates();
        let observed: HashSet<(u32, EventId)> = per_thread.keys().copied().collect();

        for (&(tid, event_id), agg) in per_thread {
            let stddev = if agg.num_samples < 2 {
                0.0
            } else {
                agg.stddev_rate
            };
            self.estimator.measurement_update(
                tid,
                event_id,
                agg.mean_rate,
                stddev,
                agg.num_samples,
                self.current_time_ns,
            );
        }

        let stale_keys: Vec<(u32, EventId)> = self
            .estimator
            .all_estimates()
            .keys()
            .filter(|k| !observed.contains(k))
            .copied()
            .collect();
        for (tid, event_id) in stale_keys {
            self.estimator.time_update(tid, event_id, elapsed_ns);
        }

        // Apply cross-event (correlated) process noise once per quantum.
        // For estimators without correlation data this is a no-op.
        let active_tids: Vec<u32> = self
            .estimator
            .all_estimates()
            .keys()
            .map(|&(tid, _)| tid)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        for tid in active_tids {
            self.estimator.quantum_step(tid, elapsed_ns);
        }
    }

    /// Borrow the state estimator for inspection (e.g. from a sink or test).
    pub fn estimator(&self) -> &dyn StateEstimator {
        self.estimator.as_ref()
    }

    /// Nanoseconds elapsed since profiling started, updated after each `step()`.
    pub fn current_time_ns(&self) -> u64 {
        self.current_time_ns
    }

    /// Total samples delivered by the source, if the source tracks it.
    pub fn samples_emitted(&self) -> Option<u64> {
        self.source.samples_emitted()
    }
}

/// Builder for `Profiler`; all four components are required before calling `build`.
pub struct ProfilerBuilder<'s> {
    /// Sample source, set via `source()`.
    source: Option<Box<dyn SampleSource>>,
    /// Initialized scheduler, set via `scheduler*()`.
    scheduler: Option<Box<dyn Scheduler>>,
    /// Output sinks slice, set via `sinks()`.
    sinks: Option<&'s mut [Box<dyn OutputSink>]>,
    /// State estimator, set via `estimator*()`.
    estimator: Option<Box<dyn StateEstimator>>,
}

impl<'s> ProfilerBuilder<'s> {
    /// Create an empty builder; all fields default to `None`.
    pub fn new() -> Self {
        Self {
            source: None,
            scheduler: None,
            sinks: None,
            estimator: None,
        }
    }

    /// Set the sample source.
    pub fn source(mut self, s: impl SampleSource + 'static) -> Self {
        self.source = Some(Box::new(s));
        self
    }

    /// Set and initialize a concrete scheduler, querying slot count from the already-set source.
    pub fn scheduler(
        mut self,
        mut s: impl Scheduler + 'static,
        all_events: Vec<EventId>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let num_slots = self
            .source
            .as_ref()
            .map(|src| src.num_slots())
            .unwrap_or(MAX_COUNTERS);
        s.init(all_events, num_slots)?;
        self.scheduler = Some(Box::new(s));
        Ok(self)
    }

    /// Set and initialize a boxed scheduler, querying slot count from the already-set source.
    pub fn scheduler_boxed(
        mut self,
        mut s: Box<dyn Scheduler>,
        all_events: Vec<EventId>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let num_slots = self
            .source
            .as_ref()
            .map(|src| src.num_slots())
            .unwrap_or(MAX_COUNTERS);
        s.init(all_events, num_slots)?;
        self.scheduler = Some(s);
        Ok(self)
    }

    /// Store a scheduler that has already been initialized. Skips the `init` call.
    pub fn scheduler_boxed_pre_init(mut self, s: Box<dyn Scheduler>) -> Self {
        self.scheduler = Some(s);
        self
    }

    /// Set the state estimator.
    pub fn estimator(mut self, e: impl StateEstimator + 'static) -> Self {
        self.estimator = Some(Box::new(e));
        self
    }

    /// Set the state estimator from an already-boxed value.
    pub fn estimator_boxed(mut self, e: Box<dyn StateEstimator>) -> Self {
        self.estimator = Some(e);
        self
    }

    /// Set the output sinks slice.
    pub fn sinks(mut self, sinks: &'s mut [Box<dyn OutputSink>]) -> Self {
        self.sinks = Some(sinks);
        self
    }

    /// Consume the builder and construct a `Profiler`. Panics if any required component is missing.
    pub fn build(self) -> Profiler<'s> {
        Profiler {
            source: self.source.expect("ProfilerBuilder: source is required"),
            scheduler: self
                .scheduler
                .expect("ProfilerBuilder: scheduler is required"),
            sinks: self.sinks.expect("ProfilerBuilder: sinks is required"),
            estimator: self
                .estimator
                .expect("ProfilerBuilder: estimator is required"),
            active_set: Vec::new(),
            current_time_ns: 0,
        }
    }
}

impl<'s> Default for ProfilerBuilder<'s> {
    fn default() -> Self {
        Self::new()
    }
}
