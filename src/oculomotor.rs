use crate::event_registry::{EventId, EventRegistry};
use crate::hardware_counters::HardwareCounters;
use crate::sampler::{self, SamplerSkelBuilder};
use crate::scheduler::ScheduleDecision;
use crate::scheduler::Scheduler;
use libbpf_rs::RingBufferBuilder;
use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};
use perf_event::{Builder, events};
use std::fs::File;
use std::io::Write;
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::hardware_counters::MAX_COUNTERS;

const TASK_COMM_LEN: usize = 16;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct SaccadeSample {
    timestamp_ns: u64,
    duration_ns: u64,
    pid: u32,
    cpu_id: u32,
    type_: u32,
    pad: u32,
    values: [u64; MAX_COUNTERS],
    task: [u8; TASK_COMM_LEN],
}

pub struct Oculomotor {
    skel: sampler::SamplerSkel<'static>,
    ringbuf: libbpf_rs::RingBuffer<'static>,
    scheduler: Box<dyn Scheduler>,
    active_set: Vec<EventId>,
    registry: EventRegistry,
    _cpus: Vec<usize>,
    _output_file: Arc<Mutex<File>>,
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
        let mut skel_builder = SamplerSkelBuilder::default();
        skel_builder.obj_builder.debug(true);

        let open_object = Box::new(MaybeUninit::uninit());
        let open_object_ref = Box::leak(open_object);
        let mut open_skel = skel_builder.open(open_object_ref)?;

        // Initialize global config before loading.
        // target_pid (0) maps to .bss.
        if let Some(bss) = open_skel.maps.bss_data.as_mut() {
            bss.target_pid = target_pid;
        }
        // min_sample_interval_ns (non-zero) maps to .data (or .bss if init was 0, but it was 1000000).
        // Let's check both or assume .data for initialized.
        if let Some(data) = open_skel.maps.data_data.as_mut() {
            data.min_sample_interval_ns = 1_000;
        }

        let mut skel = open_skel.load()?;
        skel.attach()?;

        let file = File::create(output_path)?;
        let output_file = Arc::new(Mutex::new(file));

        // Write CSV header
        {
            let mut f = output_file.lock().unwrap();
            writeln!(
                f,
                "timestamp,duration,pid,cpu_id,type,val0,val1,val2,val3,task"
            )?;
        }

        let callback_file = output_file.clone();
        let mut ringbuf_builder = RingBufferBuilder::new();
        ringbuf_builder.add(&skel.maps.ringbuf, move |data| {
            if data.len() < std::mem::size_of::<SaccadeSample>() {
                return 0;
            }
            let sample = unsafe { &*(data.as_ptr() as *const SaccadeSample) };

            let task_name = std::str::from_utf8(&sample.task)
                .unwrap_or("<unknown>")
                .trim_end_matches('\0');

            let mut f = callback_file.lock().unwrap();
            let _ = writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{}",
                sample.timestamp_ns,
                sample.duration_ns,
                sample.pid,
                sample.cpu_id,
                sample.type_,
                sample.values[0],
                sample.values[1],
                sample.values[2],
                sample.values[3],
                task_name
            );
            0
        })?;

        let ringbuf = ringbuf_builder.build()?;

        let num_cpus = std::thread::available_parallelism()?.get();
        let cpus: Vec<usize> = (0..num_cpus).collect();

        let mut timer_links = Vec::new();
        let mut timer_events = Vec::new();

        for cpu in &cpus {
            let mut counter = Builder::new(events::Software::CPU_CLOCK)
                .one_cpu(*cpu)
                .any_pid()
                .sample_frequency(1000)
                .build()?;

            counter.enable()?;

            let link = skel
                .progs
                .handle_timer
                .attach_perf_event(counter.as_raw_fd())?;
            timer_links.push(link);
            timer_events.push(counter);
        }

        let hw_counters = HardwareCounters::new(cpus.len());

        Ok(Self {
            skel,
            ringbuf,
            scheduler,
            active_set: Vec::new(),
            _cpus: cpus,
            registry,
            _output_file: output_file,
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
        self.ringbuf.poll(Duration::from_millis(1))?;
        Ok(())
    }

    pub fn update_counters(
        &mut self,
        decision: &ScheduleDecision,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let new_set = &decision.active_events;

        // We need to map the new set of events to the 4 slots.
        // Simple strategy: Just assign them in order 0..min(4, len).
        // If the event at slot i is different from what we had, update it.

        // We'll track what we *want* in each slot.
        let mut desired_events = [None; MAX_COUNTERS];
        for (i, &event_id) in new_set.iter().take(MAX_COUNTERS).enumerate() {
            desired_events[i] = Some(self.registry.get_event(event_id));
        }

        // Now iterate and update slots
        for (i, event) in desired_events.iter().enumerate() {
            let old_id = self.active_set.get(i).copied();
            let new_id = new_set.get(i).copied();

            if old_id != new_id {
                self.hw_counters
                    .update_slot(i, *event, &self.skel.maps.counters)?;
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
