pub mod buffered_output;
pub mod cli;
pub mod counter_backend;
pub mod docs;
pub mod event_library;
pub mod event_registry;
pub mod hardware_backend;
pub mod hardware_counters;
pub mod oculomotor;
pub mod perf;
pub mod perfetto_trace;
pub mod scheduler;
pub mod syscalls;
pub mod virtual_backend;
pub mod virtual_counter;

#[path = "bpf/sampler.skel.rs"]
mod sampler;
