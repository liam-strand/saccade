use crate::event_library::Event;
use crate::syscalls::{self, CpuSet};
use libbpf_rs::{Map, MapCore, MapFlags};
use perf_event::{Builder, events};
use std::os::fd::AsRawFd;

pub const MAX_COUNTERS: usize = 4;

pub struct HardwareCounters {
    num_cpus: usize,
    /// Store active counters.
    /// Index 1: Slot index (0..MAX_COUNTERS)
    /// Index 2: CPU index (0..num_cpus)
    active_counters: Vec<Vec<perf_event::Counter>>,
}

impl HardwareCounters {
    pub fn new(num_cpus: usize) -> Self {
        // Initialize with MAX_COUNTERS empty slots
        let mut active_counters = Vec::with_capacity(MAX_COUNTERS);
        for _ in 0..MAX_COUNTERS {
            active_counters.push(Vec::new());
        }

        Self {
            num_cpus,
            active_counters,
        }
    }

    /// Updates a specific counter slot with a new event (or clears it if event is None).
    pub fn update_slot(
        &mut self,
        slot_idx: usize,
        event: Option<&Event>,
        bpf_map: &Map,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let event_config = match event {
            Some(e) => e,
            None => {
                // If we are clearing the slot, we can just clear the vector.
                // The FDs will be closed and map entries removed by kernel.
                self.active_counters[slot_idx].clear();
                return Ok(());
            }
        };

        let mut new_counters = Vec::with_capacity(self.num_cpus);

        for cpu in 0..self.num_cpus {
            // On x86, encoding is usually: event (0-7), umask (8-15).
            let mut counter =
                Builder::new(events::Raw::new(event_config.event).config1(event_config.umask))
                    .one_cpu(cpu)
                    .any_pid()
                    .build()?;

            counter.enable()?;
            new_counters.push(counter);
        }

        let mut full_mask = CpuSet::new();
        for i in 0..self.num_cpus {
            full_mask.set(i);
        }

        for cpu in 0..self.num_cpus {
            let mut mask = CpuSet::new();
            mask.set(cpu);
            syscalls::sched_setaffinity(0, &mask)?;
            let _ = syscalls::sched_yield();

            // Key is the index: (cpu * MAX_COUNTERS) + slot_idx
            let fd = new_counters[cpu].as_raw_fd() as u32;
            let map_idx = (cpu * MAX_COUNTERS + slot_idx) as u32;
            let key = map_idx.to_ne_bytes();
            let val = fd.to_ne_bytes();

            match bpf_map.update(&key, &val, MapFlags::ANY) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        "[ERROR] Failed to update map slot {} for CPU {}: {}",
                        slot_idx, cpu, e
                    );
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Map update failed: {}", e),
                    )));
                }
            }
        }

        syscalls::sched_setaffinity(0, &full_mask)?;

        // Store new counters (drops old ones, closing old FDs)
        self.active_counters[slot_idx] = new_counters;

        Ok(())
    }
}
