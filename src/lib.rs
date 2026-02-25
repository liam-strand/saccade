pub mod buffered_output;
pub mod cli;
pub mod docs;
pub mod event_library;
pub mod event_registry;
pub mod hardware_counters;
pub mod oculomotor;
pub mod perf;
pub mod scheduler;
pub mod syscalls;

#[path = "bpf/sampler.skel.rs"]
mod sampler;
