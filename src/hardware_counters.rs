use crate::counter_backend::MAX_COUNTERS;
use crate::event_registry::EventRegistry;
use crate::sampler::SamplerSkel;
use libbpf_rs::{MapCore, MapFlags, MapHandle};
use perf_event::{Builder, Counter, events};
use std::os::fd::AsRawFd;

pub struct HardwareCounters {
    num_cpus: usize,
    bpf_maps: [MapHandle; MAX_COUNTERS],
    event_registry: EventRegistry,
    active_counters: Vec<Vec<Option<Counter>>>,
}

impl HardwareCounters {
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

    pub fn update_slot(
        &mut self,
        skel: &mut SamplerSkel<'static>,
        slot_idx: usize,
        event_id: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bpf_map = &self.bpf_maps[slot_idx];
        let event = self.event_registry.get_event(event_id);

        self.stop_counters(skel);

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

        Ok(())
    }

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

    fn start_counters(&self, skel: &mut SamplerSkel<'static>) {
        skel.maps.bss_data.as_mut().unwrap().tracking = true;
    }
}
