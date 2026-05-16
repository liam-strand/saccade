use crate::event::EventId;

mod wire_types {
    #![allow(non_camel_case_types, dead_code)]
    include!(concat!(env!("OUT_DIR"), "/wire_types.rs"));
}

pub use wire_types::{
    MAX_COUNTERS, MAX_CPUS, SampleType, TASK_COMM_LEN, saccade_sample as WireSample,
};

/// One observation of one hardware event from one CPU in one timeslice.
///
/// Carries the event **count** (delta since the last sample for this cpu/slot) and the
/// **duration** of the measurement interval. Rate computation (`count / duration_ns`)
/// happens downstream in `Quantum::aggregates()`.
#[derive(Debug, Clone)]
pub struct RawSample {
    /// Kernel timestamp at the end of this sample interval (nanoseconds, ktime).
    pub timestamp_ns: u64,
    /// Duration of this measurement interval in nanoseconds.
    pub duration_ns: u64,
    /// Logical CPU index on which the sample was taken.
    pub cpu_id: u32,
    /// Thread group ID (userspace PID) of the sampled task.
    pub pid: u32,
    /// Kernel thread ID of the sampled task.
    pub tid: u32,
    /// The hardware event that was counted.
    pub event_id: EventId,
    /// Delta event count since the previous sample for this (cpu, slot) pair.
    pub count: u64,
    /// Null-terminated task comm string from the kernel.
    pub task: [u8; TASK_COMM_LEN],
}
