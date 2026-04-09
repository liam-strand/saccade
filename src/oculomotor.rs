use crate::buffered_output::Logger;
use crate::counter_backend::CounterBackend;
use crate::event_registry::EventId;
use crate::perfetto::PerfettoWriter;
use crate::scheduler::Scheduler;
use crate::virtual_counter::VirtualCounterState;
use std::time::Duration;

pub struct Oculomotor {
    backend: Box<dyn CounterBackend>,
    scheduler: Box<dyn Scheduler>,
    active_set: Vec<EventId>,
    _logger: Option<Logger>,
    vcs: VirtualCounterState,
    last_step_ns: u64,
    trace_writer: Option<PerfettoWriter>,
}

impl Oculomotor {
    pub fn new(
        backend: Box<dyn CounterBackend>,
        scheduler: Box<dyn Scheduler>,
        num_events: usize,
        logger: Option<Logger>,
        trace_writer: Option<PerfettoWriter>,
    ) -> Self {
        Self {
            backend,
            scheduler,
            active_set: Vec::new(),
            _logger: logger,
            vcs: VirtualCounterState::new(num_events),
            last_step_ns: 0,
            trace_writer,
        }
    }

    pub fn step(&mut self) -> Option<Duration> {
        let observations = self.backend.poll_observations();

        let elapsed_ns = observations
            .iter()
            .map(|o| o.total_duration_ns)
            .sum::<u64>()
            .max(1);

        // Measurement update for observed counters
        let mut observed: Vec<EventId> = Vec::new();
        for obs in &observations {
            if obs.total_duration_ns > 0 {
                let rate = obs.total_count as f64 / obs.total_duration_ns as f64;
                let stddev = if obs.num_samples < 2 {
                    0.0
                } else {
                    obs.stddev_rate
                };
                self.vcs.measurement_update(
                    obs.event_id,
                    rate,
                    stddev,
                    self.last_step_ns + elapsed_ns,
                );
                observed.push(obs.event_id);
            }
        }

        // Time update (uncertainty growth) for unobserved counters
        for id in 0..self.vcs.num_events() as EventId {
            if !observed.contains(&id) {
                self.vcs.time_update(id, elapsed_ns);
            }
        }

        self.last_step_ns += elapsed_ns;

        let decision = self.scheduler.next_step(&self.vcs);
        self.backend
            .update_counters(&self.active_set, &decision.active_events)
            .unwrap();
        self.active_set = decision.active_events.clone();

        if let Some(ref mut writer) = self.trace_writer {
            let _ = writer.emit_step(self.last_step_ns, &self.vcs, &self.active_set);
        }

        decision.duration
    }

    pub fn vcs(&self) -> &VirtualCounterState {
        &self.vcs
    }

    pub fn last_step_ns(&self) -> u64 {
        self.last_step_ns
    }
}
