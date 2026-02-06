use crate::event_registry::EventId;
use crate::scheduler::ScheduleDecision;
use crate::scheduler::Scheduler;
use libbpf_rs::RingBufferBuilder;
use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};
use std::os::fd::AsRawFd;
use std::time::Duration;

#[path = "bpf/sampler.skel.rs"]
mod sampler_skel;
use sampler_skel::SamplerSkelBuilder;

const MAX_COUNTERS: usize = 4;

pub struct Oculomotor {
    skel: sampler_skel::SamplerSkel<'static>,
    ringbuf: libbpf_rs::RingBuffer<'static>,
    _scheduler: Box<dyn Scheduler>,
    active_set: Vec<EventId>,
    cpus: Vec<usize>,
    // We store counters to keep them alive.
    counters: Vec<Vec<perf_event::Counter>>,
}

impl Oculomotor {
    pub fn new(
        target_pid: u32,
        scheduler: Box<dyn Scheduler>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut skel_builder = SamplerSkelBuilder::default();
        skel_builder.obj_builder.debug(true);

        let mut open_skel = skel_builder.open()?;

        // Initialize global config before loading.
        // target_pid (0) maps to .bss.
        if let Some(bss) = open_skel.maps.bss_data.as_mut() {
            bss.target_pid = target_pid;
        }
        // min_sample_interval_ns (non-zero) maps to .data (or .bss if init was 0, but it was 1000000).
        // Let's check both or assume .data for initialized.
        if let Some(data) = open_skel.maps.data_data.as_mut() {
            data.min_sample_interval_ns = 1_000_000;
        }

        let mut skel = open_skel.load()?;
        skel.attach()?;

        let ringbuf_builder = RingBufferBuilder::new();
        let ringbuf = ringbuf_builder
            .add(skel.maps().ringbuf(), |data| {
                // TODO: Handle sample data properly (decode Sample struct)
                0
            })?
            .build()?;

        let num_cpus = std::thread::available_parallelism()?.get();
        let cpus = (0..num_cpus).collect();

        Ok(Self {
            skel,
            ringbuf,
            _scheduler: scheduler,
            active_set: Vec::new(),
            cpus,
            counters: Vec::new(),
        })
    }

    pub fn set_target_pid(&mut self, pid: u32) {
        if let Some(bss) = self.skel.maps_mut().bss_data.as_mut() {
            bss.target_pid = pid;
        }
    }

    pub fn set_sample_rate(&mut self, interval_ns: u64) {
        if let Some(data) = self.skel.maps_mut().data_data.as_mut() {
            data.min_sample_interval_ns = interval_ns;
        }
    }

    pub fn poll(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.ringbuf.poll(Duration::from_millis(10))?;
        Ok(())
    }

    pub fn update_counters(
        &mut self,
        decision: &ScheduleDecision,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1. Teardown: Simply clearing the vector drops the Counters, closing FDs.
        self.counters.clear();

        // 2. Setup: Open new counters
        for (slot_idx, &event_id) in decision.active_events.iter().take(MAX_COUNTERS).enumerate() {
            let mut cpu_counters = Vec::new();

            // Placeholder: Assuming INSTRUCTIONS for testing.
            // Ideally should use Registry to get config.

            for cpu in &self.cpus {
                let mut builder = perf_event::Builder::new();
                builder.kind(perf_event::events::Hardware::INSTRUCTIONS);
                builder.observe_cpu(*cpu);

                let counter = builder.build()?;
                counter.enable()?;

                // TODO: Update BPF map with counter.as_raw_fd()
                // For now, we just open them.

                cpu_counters.push(counter);
            }
            self.counters.push(cpu_counters);
        }
        Ok(())
    }
}
