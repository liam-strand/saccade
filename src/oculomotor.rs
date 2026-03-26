use crate::buffered_output::Logger;
use crate::event_registry::{EventId, EventRegistry};
use crate::hardware_counters::HardwareCounters;
use crate::sampler::{self, SamplerSkelBuilder};
use crate::scheduler::ScheduleDecision;
use crate::scheduler::Scheduler;
use libbpf_rs::RingBufferBuilder;
use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};
use perf_event::{Builder, events};
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::time::Duration;

use crate::hardware_counters::MAX_COUNTERS;

pub const TASK_COMM_LEN: usize = 16;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct SaccadeSample {
    pub timestamp_ns: u64,
    pub duration_ns: u64,
    pub pid: u32,
    pub cpu_id: u32,
    pub type_: u32,
    pub pad: u32,
    pub values: [u64; MAX_COUNTERS],
    pub events: [u64; MAX_COUNTERS],
    pub task: [u8; TASK_COMM_LEN],
}

pub struct Oculomotor {
    skel: sampler::SamplerSkel<'static>,
    ringbuf: libbpf_rs::RingBuffer<'static>,
    scheduler: Box<dyn Scheduler>,
    active_set: Vec<EventId>,
    _cpus: Vec<usize>,
    _logger: Logger,
    hw_counters: HardwareCounters,
    _timer_links: Vec<libbpf_rs::Link>,
    _timer_events: Vec<perf_event::Counter>,
}

impl Oculomotor {
    pub fn new(
        target_pid: u32,
        registry: EventRegistry,
        scheduler: Box<dyn Scheduler>,
        output_path: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let skel_builder = SamplerSkelBuilder::default();

        let open_object = Box::new(MaybeUninit::uninit());
        let open_object_ref = Box::leak(open_object);
        let mut open_skel = skel_builder.open(open_object_ref)?;

        open_skel
            .maps
            .bss_data
            .as_mut()
            .expect("Failed to set target PID")
            .target_pid = target_pid;
        open_skel
            .maps
            .data_data
            .as_mut()
            .expect("Failed to set min sample interval")
            .min_sample_interval_ns = 100000;

        let mut skel = open_skel.load()?;
        skel.attach()?;

        let logger = Logger::new(output_path, 256_000)?;
        let sender = logger.clone_sender().expect("Failed to get logger sender");

        eprintln!("Oculomotor attached to PID {}", target_pid);

        let mut ringbuf_builder = RingBufferBuilder::new();
        ringbuf_builder.add(&skel.maps.ringbuf, move |data| {
            let sample = unsafe { *(data.as_ptr() as *const SaccadeSample) };
            let _ = sender.try_send(sample);
            0
        })?;

        let ringbuf = ringbuf_builder.build()?;

        let num_cpus = std::thread::available_parallelism()?.get();
        let cpus: Vec<usize> = (0..num_cpus).collect();

        eprintln!("Oculomotor has {} CPUs", num_cpus);

        let mut timer_links = Vec::new();
        let mut timer_events = Vec::new();

        for cpu in &cpus {
            let mut counter = Builder::new(events::Software::CPU_CLOCK)
                .one_cpu(*cpu)
                .any_pid()
                .sample_frequency(15000)
                .build()?;

            counter.enable()?;

            let link = skel
                .progs
                .handle_timer
                .attach_perf_event(counter.as_raw_fd())?;
            timer_links.push(link);
            timer_events.push(counter);
        }

        eprintln!("Oculomotor has {} timer events", timer_events.len());

        let hw_counters = HardwareCounters::new(cpus.len(), registry, &mut skel);

        eprintln!("Hardware counters initialized");

        Ok(Self {
            skel,
            ringbuf,
            scheduler,
            active_set: Vec::new(),
            _cpus: cpus,
            _logger: logger,
            _timer_links: timer_links,
            _timer_events: timer_events,
            hw_counters,
        })
    }

    pub fn set_target_pid(&mut self, pid: u32) {
        if let Some(bss) = self.skel.maps.bss_data.as_mut() {
            bss.target_pid = pid;
        }
    }

    pub fn set_sample_rate(&mut self, interval_ns: u64) {
        if let Some(data) = self.skel.maps.data_data.as_mut() {
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
        let new_set = &decision.active_events;

        if self.active_set.is_empty() {
            for (i, &id) in new_set.iter().enumerate() {
                self.hw_counters.update_slot(&mut self.skel, i, id)?;
            }
        } else {
            for (i, &old_id) in self.active_set.iter().enumerate() {
                if old_id != new_set[i] {
                    self.hw_counters
                        .update_slot(&mut self.skel, i, new_set[i])?;
                }
            }
        }

        self.active_set = new_set.clone();
        Ok(())
    }

    pub fn step(&mut self) -> Option<Duration> {
        self.poll().unwrap();
        let decision = self.scheduler.next_step();
        self.update_counters(&decision).unwrap();
        decision.duration
    }
}
