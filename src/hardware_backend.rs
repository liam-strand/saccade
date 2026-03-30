use crate::counter_backend::{CounterBackend, MAX_COUNTERS, Observation, SaccadeSample};
use crate::event_registry::{EventId, EventRegistry};
use crate::hardware_counters::HardwareCounters;
use crate::sampler::SamplerSkelBuilder;
use libbpf_rs::RingBufferBuilder;
use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};
use perf_event::{Builder, events};
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::Duration;
use tracing::debug;

pub struct HardwareBackend {
    skel: crate::sampler::SamplerSkel<'static>,
    ringbuf: libbpf_rs::RingBuffer<'static>,
    hw_counters: HardwareCounters,
    _timer_links: Vec<libbpf_rs::Link>,
    _timer_events: Vec<perf_event::Counter>,
    observation_rx: Receiver<SaccadeSample>,
}

impl HardwareBackend {
    pub fn new(
        target_pid: u32,
        registry: EventRegistry,
        logger_tx: SyncSender<SaccadeSample>,
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

        debug!("HardwareBackend attached to PID {}", target_pid);

        let (obs_tx, obs_rx) = mpsc::sync_channel::<SaccadeSample>(256_000);

        let mut ringbuf_builder = RingBufferBuilder::new();
        ringbuf_builder.add(&skel.maps.ringbuf, move |data| {
            let sample = unsafe { *(data.as_ptr() as *const SaccadeSample) };
            let _ = logger_tx.try_send(sample);
            let _ = obs_tx.try_send(sample);
            0
        })?;

        let ringbuf = ringbuf_builder.build()?;

        let num_cpus = std::thread::available_parallelism()?.get();
        let cpus: Vec<usize> = (0..num_cpus).collect();

        debug!("HardwareBackend has {} CPUs", num_cpus);

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

        debug!("HardwareBackend has {} timer events", timer_events.len());

        let hw_counters = HardwareCounters::new(cpus.len(), registry, &mut skel);

        debug!("Hardware counters initialized");

        Ok(Self {
            skel,
            ringbuf,
            hw_counters,
            _timer_links: timer_links,
            _timer_events: timer_events,
            observation_rx: obs_rx,
        })
    }
}

impl CounterBackend for HardwareBackend {
    fn poll_observations(&mut self) -> Vec<Observation> {
        let _ = self.ringbuf.poll(Duration::from_millis(10));

        let mut by_event: HashMap<EventId, Observation> = HashMap::new();

        while let Ok(sample) = self.observation_rx.try_recv() {
            for slot in 0..MAX_COUNTERS {
                let event_id = sample.events[slot] as EventId;
                let value = sample.values[slot];
                if value == 0 && sample.events[slot] == 0 {
                    continue;
                }
                let obs = by_event.entry(event_id).or_insert(Observation {
                    event_id,
                    total_count: 0,
                    total_duration_ns: 0,
                });
                obs.total_count += value;
                obs.total_duration_ns += sample.duration_ns;
            }
        }

        by_event.into_values().collect()
    }

    fn update_counters(
        &mut self,
        old_set: &[EventId],
        new_set: &[EventId],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if old_set.is_empty() {
            for (i, &id) in new_set.iter().enumerate() {
                self.hw_counters.update_slot(&mut self.skel, i, id)?;
            }
        } else {
            for (i, &old_id) in old_set.iter().enumerate() {
                if old_id != new_set[i] {
                    self.hw_counters
                        .update_slot(&mut self.skel, i, new_set[i])?;
                }
            }
        }
        Ok(())
    }
}
