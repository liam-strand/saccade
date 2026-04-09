use crate::event_registry::{EventId, EventRegistry};
use crate::hardware_counters::HardwareCounters;
use crate::sample::{MAX_COUNTERS, MAX_CPUS, RawSample, SampleType, WireSample};
use crate::sampler::SamplerSkelBuilder;
use crate::source::SampleSource;
use libbpf_rs::RingBufferBuilder;
use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};
use perf_event::{Builder, events};
use std::mem::MaybeUninit;
use std::os::fd::AsRawFd;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::{Duration, Instant};
use tracing::debug;

/// eBPF-backed sample source.
///
/// Collects absolute counter readings from the BPF ringbuffer, computes
/// per-(cpu,slot) deltas in userspace, and returns typed `RawSample` values.
pub struct HardwareSampleSource {
    skel: crate::sampler::SamplerSkel<'static>,
    ringbuf: libbpf_rs::RingBuffer<'static>,
    hw_counters: HardwareCounters,
    _timer_links: Vec<libbpf_rs::Link>,
    _timer_events: Vec<perf_event::Counter>,
    wire_rx: Receiver<WireSample>,
    /// Optional forwarding channel for raw wire samples (e.g. to CSV logger).
    logger_tx: Option<SyncSender<WireSample>>,
    /// Per-(cpu, slot) baseline absolute counter values.
    /// Updated on every sample; reset on RESUME markers.
    baselines: [[u64; MAX_COUNTERS]; MAX_CPUS],
    last_collect: Instant,
}

impl HardwareSampleSource {
    pub fn new(
        target_pid: u32,
        registry: EventRegistry,
        logger_tx: Option<SyncSender<WireSample>>,
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
            .min_sample_interval_ns = 100_000;

        let mut skel = open_skel.load()?;
        skel.attach()?;

        debug!("HardwareSampleSource attached to PID {}", target_pid);

        let (wire_tx, wire_rx) = mpsc::sync_channel::<WireSample>(256_000);

        let mut ringbuf_builder = RingBufferBuilder::new();
        ringbuf_builder.add(&skel.maps.ringbuf, move |data| {
            let wire = unsafe { *(data.as_ptr() as *const WireSample) };
            let _ = wire_tx.try_send(wire);
            0
        })?;

        let ringbuf = ringbuf_builder.build()?;

        let num_cpus = std::thread::available_parallelism()?.get();
        let cpus: Vec<usize> = (0..num_cpus).collect();

        debug!("HardwareSampleSource has {} CPUs", num_cpus);

        let mut timer_links = Vec::new();
        let mut timer_events = Vec::new();

        for cpu in &cpus {
            let mut counter = Builder::new(events::Software::CPU_CLOCK)
                .one_cpu(*cpu)
                .any_pid()
                .sample_frequency(15_000)
                .build()?;

            counter.enable()?;

            let link = skel
                .progs
                .handle_timer
                .attach_perf_event(counter.as_raw_fd())?;
            timer_links.push(link);
            timer_events.push(counter);
        }

        debug!("HardwareSampleSource has {} timer events", timer_events.len());

        let hw_counters = HardwareCounters::new(cpus.len(), registry, &mut skel);

        debug!("Hardware counters initialized");

        Ok(Self {
            skel,
            ringbuf,
            hw_counters,
            _timer_links: timer_links,
            _timer_events: timer_events,
            wire_rx,
            logger_tx,
            baselines: [[0u64; MAX_COUNTERS]; MAX_CPUS],
            last_collect: Instant::now(),
        })
    }

    /// Convert a `WireSample` (absolute counter readings) to per-event `RawSample`s
    /// (delta counts). Updates per-(cpu,slot) baselines. RESUME markers reset baselines
    /// without producing output samples.
    fn wire_to_raw(&mut self, wire: &WireSample) -> Vec<RawSample> {
        let cpu = wire.cpu_id as usize;
        if cpu >= MAX_CPUS {
            return Vec::new();
        }

        // RESUME marker: reset baselines, emit no samples
        if wire.type_ == SampleType::Resume as u32 {
            for slot in 0..MAX_COUNTERS {
                self.baselines[cpu][slot] = wire.counters[slot];
            }
            return Vec::new();
        }

        if wire.duration_ns == 0 {
            return Vec::new();
        }

        let mut samples = Vec::with_capacity(MAX_COUNTERS);
        for slot in 0..MAX_COUNTERS {
            let event_id = wire.events[slot] as EventId;
            // Skip slots with no event assigned
            if event_id == 0 && wire.events[slot] == 0 {
                continue;
            }

            let abs_value = wire.counters[slot];
            let baseline = self.baselines[cpu][slot];
            // If counter rolled back (e.g. slot was just reassigned), treat as 0.
            let count = abs_value.saturating_sub(baseline);
            self.baselines[cpu][slot] = abs_value;

            samples.push(RawSample {
                timestamp_ns: wire.timestamp_ns,
                duration_ns: wire.duration_ns,
                cpu_id: wire.cpu_id,
                pid: wire.pid,
                event_id,
                count,
                task: wire.task,
            });
        }
        samples
    }
}

impl SampleSource for HardwareSampleSource {
    fn collect(&mut self) -> (Vec<RawSample>, u64) {
        let _ = self.ringbuf.poll(Duration::from_millis(10));

        let elapsed_ns = self.last_collect.elapsed().as_nanos() as u64;
        self.last_collect = Instant::now();

        let mut raw_samples = Vec::new();
        while let Ok(wire) = self.wire_rx.try_recv() {
            if let Some(tx) = &self.logger_tx {
                let _ = tx.try_send(wire);
            }
            raw_samples.extend(self.wire_to_raw(&wire));
        }

        (raw_samples, elapsed_ns.max(1))
    }

    fn apply_schedule(
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
        // When counters are swapped, baselines are stale — reset them all.
        // The next sample from each CPU will establish a new baseline via saturating_sub.
        self.baselines = [[0u64; MAX_COUNTERS]; MAX_CPUS];
        Ok(())
    }

    fn num_slots(&self) -> usize {
        MAX_COUNTERS
    }
}
