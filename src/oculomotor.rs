use crate::event_registry::{EventId, EventRegistry};
use crate::sampler::{self, SamplerSkelBuilder};
use crate::scheduler::ScheduleDecision;
use crate::scheduler::Scheduler;
use libbpf_rs::RingBufferBuilder;
use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};
use std::fs::File;
use std::io::Write;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MAX_COUNTERS: usize = 4;
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
    cpus: Vec<usize>,
    // We store counters to keep them alive.
    counters: Vec<Vec<perf_event::Counter>>,
    output_file: Arc<Mutex<File>>,
}

impl Oculomotor {
    pub fn new(
        target_pid: u32,
        registry: EventRegistry,
        scheduler: Box<dyn Scheduler>,
        output_path: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut skel_builder = SamplerSkelBuilder::default();
        // skel_builder.obj_builder.debug(true);

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
            data.min_sample_interval_ns = 1_000_000;
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
        let cpus = (0..num_cpus).collect();

        Ok(Self {
            skel,
            ringbuf,
            scheduler,
            active_set: Vec::new(),
            cpus,
            registry,
            counters: Vec::new(),
            output_file,
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
        // 1. Teardown: Simply clearing the vector drops the Counters, closing FDs.
        self.counters.clear();

        // 2. Setup: Open new counters
        for (_slot_idx, &_event_id) in decision.active_events.iter().take(MAX_COUNTERS).enumerate()
        {
            let cpu_counters = Vec::new();

            // Placeholder: Assuming INSTRUCTIONS for testing.
            // Ideally should use Registry to get config.

            for _cpu in &self.cpus {
                // let mut builder = perf_event::Builder::new();
                // builder.kind(perf_event::events::Hardware::INSTRUCTIONS);
                // builder.observe_cpu(*cpu);

                // let mut counter = builder.build()?;
                // counter.enable()?;

                // TODO: Update BPF map with counter.as_raw_fd()
                // For now, we just open them.

                // cpu_counters.push(counter);
            }
            self.counters.push(cpu_counters);
        }
        Ok(())
    }

    pub fn step(&mut self) -> Option<Duration> {
        self.poll().unwrap();
        let decision = self.scheduler.next_step();
        self.update_counters(&decision).unwrap();
        decision.duration
    }
}
