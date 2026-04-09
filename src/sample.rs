use crate::event::EventId;

/// Must match `enum SampleType` in `sampler.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SampleType {
    Intermediate = 0,
    Flush = 1,
    /// Baseline reset marker — not a real measurement. Userspace uses the
    /// counter values to update per-(cpu,slot) baselines; no `RawSample` is emitted.
    Resume = 2,
}

pub const MAX_COUNTERS: usize = 4;
pub const MAX_CPUS: usize = 256;
pub const TASK_COMM_LEN: usize = 16;

/// eBPF wire format. `repr(C)` must match `sampler.h` exactly.
/// Used only at the eBPF boundary; convert immediately to `RawSample` before any further processing.
///
/// `counters` carries **absolute** perf counter readings. Delta computation (baseline subtraction)
/// happens in `HardwareSampleSource::wire_to_raw()` in userspace.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct WireSample {
    pub timestamp_ns: u64,
    pub duration_ns: u64,
    pub pid: u32,
    pub cpu_id: u32,
    pub type_: u32,
    pub pad: u32,
    pub counters: [u64; MAX_COUNTERS], // absolute perf counter readings
    pub events: [u64; MAX_COUNTERS],   // active event IDs per slot
    pub task: [u8; TASK_COMM_LEN],
}

/// One observation of one hardware event from one CPU in one timeslice.
///
/// Carries the event **count** (delta since the last sample for this cpu/slot) and the
/// **duration** of the measurement interval. Does **not** contain a rate — rate computation
/// (`count / duration_ns`) happens downstream in `Quantum::aggregates()`.
#[derive(Debug, Clone)]
pub struct RawSample {
    /// Kernel timestamp at the end of this sample interval (nanoseconds, ktime).
    pub timestamp_ns: u64,
    /// Duration of this measurement interval in nanoseconds.
    pub duration_ns: u64,
    pub cpu_id: u32,
    pub pid: u32,
    /// The hardware event that was counted.
    pub event_id: EventId,
    /// Number of events that occurred during `duration_ns`. This is a delta, not an absolute value.
    pub count: u64,
    /// Null-terminated task comm string from the kernel.
    pub task: [u8; TASK_COMM_LEN],
}
