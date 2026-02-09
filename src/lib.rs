pub mod cli;
pub mod docs;
pub mod event_library;
pub mod event_registry;
pub mod oculomotor;
pub mod perf;
pub mod scheduler;

#[path = "bpf/sampler.skel.rs"]
mod sampler;
