use crate::event::EventId;
use crate::quantum::Quantum;
use crate::scheduler::Scheduler;
use crate::sink::OutputSink;
use crate::source::SampleSource;
use crate::state::StateEstimator;
use crate::state::ema::VirtualCounterState;
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
pub struct Profiler {
    source: Box<dyn SampleSource>,
    scheduler: Box<dyn Scheduler>,
    sinks: Vec<Box<dyn OutputSink>>,
    estimator: Box<dyn StateEstimator>,
    active_set: Vec<EventId>,
    current_time_ns: u64,
}

impl Profiler {
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
        self.source
            .apply_schedule(&self.active_set, &decision.active_events)
            .unwrap();
        self.active_set = decision.active_events;

        // 6. Emit to all sinks
        for sink in &mut self.sinks {
            let _ = sink.emit(&quantum, self.estimator.as_ref(), &self.active_set);
        }

        decision.duration
    }

    fn update_estimator(&mut self, quantum: &Quantum, elapsed_ns: u64) {
        let aggregates = quantum.aggregates();
        let observed: Vec<EventId> = aggregates.keys().copied().collect();

        for (&event_id, agg) in aggregates {
            let stddev = if agg.num_samples < 2 {
                0.0
            } else {
                agg.stddev_rate
            };
            self.estimator.measurement_update(
                event_id,
                agg.mean_rate,
                stddev,
                agg.num_samples,
                self.current_time_ns,
            );
        }

        for id in 0..self.estimator.num_events() as EventId {
            if !observed.contains(&id) {
                self.estimator.time_update(id, elapsed_ns);
            }
        }
    }

    pub fn estimator(&self) -> &dyn StateEstimator {
        self.estimator.as_ref()
    }

    pub fn current_time_ns(&self) -> u64 {
        self.current_time_ns
    }

    pub fn finish_sinks(&mut self) {
        for sink in &mut self.sinks {
            let _ = sink.finish();
        }
    }
}

/// Builder for `Profiler`.
pub struct ProfilerBuilder {
    source: Option<Box<dyn SampleSource>>,
    scheduler: Option<Box<dyn Scheduler>>,
    sinks: Vec<Box<dyn OutputSink>>,
    estimator: Option<Box<dyn StateEstimator>>,
    num_events: usize,
}

impl ProfilerBuilder {
    pub fn new() -> Self {
        Self {
            source: None,
            scheduler: None,
            sinks: Vec::new(),
            estimator: None,
            num_events: 0,
        }
    }

    pub fn source(mut self, s: impl SampleSource + 'static) -> Self {
        self.source = Some(Box::new(s));
        self
    }

    pub fn scheduler(mut self, mut s: impl Scheduler + 'static, all_events: Vec<EventId>) -> Self {
        let num_slots = self.source.as_ref().map(|src| src.num_slots()).unwrap_or(4);
        s.init(all_events, num_slots);
        self.scheduler = Some(Box::new(s));
        self
    }

    pub fn estimator(mut self, e: impl StateEstimator + 'static) -> Self {
        self.estimator = Some(Box::new(e));
        self
    }

    pub fn add_sink(mut self, s: impl OutputSink + 'static) -> Self {
        self.sinks.push(Box::new(s));
        self
    }

    pub fn num_events(mut self, n: usize) -> Self {
        self.num_events = n;
        self
    }

    pub fn build(self) -> Profiler {
        let mut estimator: Box<dyn StateEstimator> = self
            .estimator
            .unwrap_or_else(|| Box::new(VirtualCounterState::new(0)));
        estimator.init(self.num_events);
        Profiler {
            source: self.source.expect("ProfilerBuilder: source is required"),
            scheduler: self
                .scheduler
                .expect("ProfilerBuilder: scheduler is required"),
            sinks: self.sinks,
            estimator,
            active_set: Vec::new(),
            current_time_ns: 0,
        }
    }
}

impl Default for ProfilerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
