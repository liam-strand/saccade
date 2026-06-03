//! Management of Linux perf event file descriptors and the BPF perf-event-array maps that expose
//! them to the eBPF program.

use crate::event::EventRegistry;
use crate::sample::MAX_COUNTERS;
use crate::sampler::SamplerSkel;
use libbpf_rs::{MapCore, MapFlags, MapHandle};
use perf_event::{Builder, Counter, events};
use std::os::fd::AsRawFd;

/// Owns the `perf_event` file descriptors for all counter slots across all CPUs and keeps the
/// corresponding BPF `PERF_EVENT_ARRAY` maps in sync.
pub struct HardwareCounters {
    /// Number of logical CPUs being monitored.
    num_cpus: usize,
    /// Owned handles to the four BPF `PERF_EVENT_ARRAY` maps (`counter0`–`counter3`).
    bpf_maps: [MapHandle; MAX_COUNTERS],
    /// Registry used to look up raw event/umask encodings by event ID.
    event_registry: EventRegistry,
    /// `active_counters[slot][cpu]` holds the open `Counter` for that (slot, cpu) pair, if any.
    active_counters: Vec<Vec<Option<Counter>>>,
}

impl HardwareCounters {
    /// Open BPF map handles from `skel` and allocate empty counter slots for all CPUs.
    pub fn new(
        num_cpus: usize,
        event_registry: EventRegistry,
        skel: &mut SamplerSkel<'static>,
    ) -> Self {
        let bpf_maps = [
            MapHandle::try_from(&skel.maps.counter0).expect("Failed to get counter0"),
            MapHandle::try_from(&skel.maps.counter1).expect("Failed to get counter1"),
            MapHandle::try_from(&skel.maps.counter2).expect("Failed to get counter2"),
            MapHandle::try_from(&skel.maps.counter3).expect("Failed to get counter3"),
        ];

        Self {
            num_cpus,
            bpf_maps,
            event_registry,
            active_counters: std::iter::repeat_with(|| {
                std::iter::repeat_with(|| None).take(num_cpus).collect()
            })
            .take(MAX_COUNTERS)
            .collect(),
        }
    }

    /// Replace the hardware event measured in `slot_idx` with `event_id` across all CPUs.
    ///
    /// Pauses eBPF tracking, opens fresh `perf_event` FDs for every CPU, updates the BPF map,
    /// then resumes tracking.
    ///
    /// Returns `(quiesce_ns, reconfig_ns)`: the time spent in the `stop_counters` spin-wait
    /// versus the actual `perf_event_open`/map-update reconfiguration.
    pub fn update_slot(
        &mut self,
        skel: &mut SamplerSkel<'static>,
        slot_idx: usize,
        event_id: u32,
    ) -> Result<(u64, u64), Box<dyn std::error::Error>> {
        let bpf_map = &self.bpf_maps[slot_idx];
        let event = self.event_registry.get_event(event_id);

        let quiesce_start = std::time::Instant::now();
        self.stop_counters(skel);
        let quiesce_ns = quiesce_start.elapsed().as_nanos() as u64;

        let reconfig_start = std::time::Instant::now();

        self.active_counters[slot_idx]
            .iter_mut()
            .take(self.num_cpus)
            .for_each(|slot| {
                slot.as_mut().map(|c| c.disable());
            });

        skel.maps.bss_data.as_mut().unwrap().active_counter_ids[slot_idx] = event_id;

        self.active_counters[slot_idx]
            .iter_mut()
            .take(self.num_cpus)
            .enumerate()
            .for_each(|(cpu, counter)| {
                let mut new_counter =
                    Builder::new(events::Raw::new(event.event).config1(event.umask))
                        .one_cpu(cpu)
                        .any_pid()
                        .build()
                        .expect("Failed to build counter");

                new_counter.enable().unwrap();

                let new_fd = new_counter.as_raw_fd();

                bpf_map
                    .update(
                        &(cpu as u32).to_ne_bytes(),
                        &new_fd.to_ne_bytes(),
                        MapFlags::ANY,
                    )
                    .expect("Failed to update map");

                *counter = Some(new_counter);
            });

        self.start_counters(skel);
        let reconfig_ns = reconfig_start.elapsed().as_nanos() as u64;

        Ok((quiesce_ns, reconfig_ns))
    }

    /// Signal the eBPF program to stop sampling and spin-wait until all CPUs have acknowledged.
    fn stop_counters(&self, skel: &mut SamplerSkel<'static>) {
        skel.maps.bss_data.as_mut().unwrap().tracking = false;

        while skel
            .maps
            .data_data
            .as_ref()
            .unwrap()
            .stopped
            .iter()
            .take(self.num_cpus)
            .any(|e| !e)
        {}
    }

    /// Signal the eBPF program to resume sampling.
    fn start_counters(&self, skel: &mut SamplerSkel<'static>) {
        skel.maps.bss_data.as_mut().unwrap().tracking = true;
    }
}
