//! Saccade: an eBPF-based Linux performance profiler that intelligently rotates
//! hardware counter slots across quanta to estimate rates for more counters than
//! the hardware can measure simultaneously.

/// Command-line argument types parsed by clap.
pub mod cli;
/// Top-level subcommand implementations (`run`, `sweep`, `simulate`, `evaluate`, `generate`).
pub mod commands;
/// Configuration loading, merging CLI overrides, and scheduler/estimator kind enums.
pub mod config;
/// Internal design-document modules (not part of the public API).
pub mod docs;
/// Hardware event library: `perf list` parser, `EventLibrary`, and `EventRegistry` lookup.
pub mod event;
/// eBPF map management and `perf_event` counter lifecycle (open, enable, rotate, close).
pub mod hardware_counters;
/// LLM client and prompt builder for AI-guided scheduler policies.
pub mod llm;
/// Thin wrapper around the `perf list --details` subprocess.
pub mod perf;
/// Perfetto trace writer and rate-time-series reader for `.perfetto` files.
pub mod perfetto;
/// `Profiler` orchestrator: collects samples, updates estimator, invokes scheduler, fans out to sinks.
pub mod profiler;
/// `Quantum`: per-step bundle of raw samples with lazily computed per-event aggregates.
pub mod quantum;
/// `RawSample` and wire-format types shared between eBPF and userspace.
pub mod sample;
/// Pluggable counter-rotation policies (`Scheduler` trait and built-in implementations).
pub mod scheduler;
/// Output consumers (`OutputSink` trait) including CSV, Perfetto, and HDF5 matrix sinks.
pub mod sink;
/// `SampleSource` trait and implementations (hardware eBPF source and virtual replay source).
pub mod source;
/// `StateEstimator` trait and implementations (EMA, Kalman filter, propagate).
pub mod state;
/// Safe wrappers around raw Linux syscalls used for process tracing (`ptrace`).
pub mod syscalls;

/// Auto-generated libbpf skeleton for `sampler.bpf.c` — do not edit by hand.
#[path = "bpf/sampler.skel.rs"]
mod sampler;
