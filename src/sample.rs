use crate::event::EventId;

/// Reason a sample record was emitted by the eBPF program; must match `enum SampleType` in `sampler.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SampleType {
    /// Periodic timer-fired sample while a task remains on-CPU.
    Intermediate = 0,
    /// Context-switch-out sample capturing the task's final on-CPU interval.
    Flush = 1,
    /// Baseline reset marker emitted when a CPU resumes from stopped state — not a real measurement.
    /// Userspace uses the counter values to update per-(cpu,slot) baselines; no `RawSample` is emitted.
    Resume = 2,
}

/// Maximum number of hardware perf counter slots tracked simultaneously.
pub const MAX_COUNTERS: usize = 4;
/// Maximum number of CPUs supported by the eBPF program.
pub const MAX_CPUS: usize = 256;
/// Length of the kernel task comm string, including the null terminator.
pub const TASK_COMM_LEN: usize = 16;

/// Raw eBPF ring-buffer record; `repr(C)` layout must match `struct saccade_sample` in `sampler.h` exactly.
///
/// `counters` holds **absolute** perf counter readings; delta computation happens in userspace.
/// Convert to `RawSample` immediately after reading from the ring buffer.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct WireSample {
    /// Kernel monotonic timestamp at the end of the sample interval (nanoseconds).
    pub timestamp_ns: u64,
    /// Duration of the measurement interval in nanoseconds (`duration_ns=0` for `Resume` records).
    pub duration_ns: u64,
    /// Thread group ID (userspace PID) of the sampled task.
    pub pid: u32,
    /// Logical CPU index on which the sample was taken.
    pub cpu_id: u32,
    /// Discriminant identifying why this record was emitted; corresponds to `SampleType`.
    pub type_: u32,
    /// Kernel thread ID (`task_struct->pid`) of the sampled task.
    pub tid: u32,
    /// Absolute perf counter readings for each slot; not deltas.
    pub counters: [u64; MAX_COUNTERS],
    /// Event ID active in each counter slot at sample time.
    pub events: [u64; MAX_COUNTERS],
    /// Null-terminated task comm string copied from the kernel.
    pub task: [u8; TASK_COMM_LEN],
}

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
