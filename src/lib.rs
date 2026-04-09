pub mod cli;
pub mod counter_backend; // transitional: SaccadeSample alias, Observation; remove after full migration
pub mod docs;
pub mod event;
pub mod event_library; // legacy: will merge fully into event.rs
pub mod event_registry; // legacy: will merge fully into event.rs
pub mod hardware_counters;
pub mod perf;
pub mod perfetto;
pub mod profiler;
pub mod quantum;
pub mod sample;
pub mod scheduler;
pub mod sink;
pub mod source;
pub mod syscalls;
pub mod virtual_backend; // legacy: TimeVaryingRates still used; VirtualBackend to be removed
pub mod virtual_counter;

#[path = "bpf/sampler.skel.rs"]
mod sampler;
