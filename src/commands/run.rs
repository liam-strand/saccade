use crate::commands::{load_library, spawn_child};
use crate::config::ResolvedConfig;
use crate::event::EventRegistry;
use crate::profiler::ProfilerBuilder;
use crate::sample::MAX_COUNTERS;
use crate::sink::csv::CsvSink;
use crate::sink::perfetto::PerfettoSink;
use crate::sink::{self, OutputSink};
use crate::source::hardware::HardwareSampleSource;
use crate::syscalls;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tracing::debug;

pub fn run(
    library: Option<PathBuf>,
    config: ResolvedConfig,
    trace: PathBuf,
    csv: Option<PathBuf>,
    target: Vec<String>,
) -> std::io::Result<()> {
    let lib = load_library(library)?;
    let registry = EventRegistry::new(lib);
    let all_ids = registry.get_event_ids();
    let event_names: Vec<String> = all_ids
        .iter()
        .map(|&id| registry.get_event_name(id).to_string())
        .collect();
    debug!("Loaded {} events.", all_ids.len());

    // Initialize the scheduler before spawning the child so any blocking work
    // (e.g. LLM calls) completes before the child is held in ptrace-stop.
    let mut scheduler = config.build_scheduler(&registry);
    scheduler
        .init(all_ids.clone(), MAX_COUNTERS)
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let estimator = config.build_estimator(&registry);

    let mut child = spawn_child(&target)?;
    let pid = child.id();
    syscalls::wait_for_exec(pid)?;

    let source = HardwareSampleSource::new(pid, registry, None, config.q_sample_ns)
        .expect("Failed to create hardware source");

    let mut sinks: Vec<Box<dyn OutputSink>> = Vec::new();
    sinks.push(Box::new(PerfettoSink::new(
        trace,
        event_names,
        config.q_output_ns,
    )?));
    if let Some(path) = csv {
        sinks.push(Box::new(CsvSink::new(path)?));
    }

    let mut profiler = ProfilerBuilder::new()
        .source(source)
        .scheduler_boxed_pre_init(scheduler)
        .estimator_boxed(estimator)
        .sinks(&mut sinks)
        .build();

    debug!("Profiler is ready.");
    syscalls::ptrace_detach(pid)?;

    let mut quantum_dur = Duration::from_nanos(config.q_schedule_ns);
    let mut loops = 0;
    while child
        .try_wait()
        .expect("Failed to wait for child")
        .is_none()
    {
        if let Some(d) = profiler.step() {
            quantum_dur = d;
        }
        thread::sleep(quantum_dur);
        loops += 1;
    }
    child.wait().unwrap();
    drop(profiler);
    sink::finish_sinks(&mut sinks);
    debug!("Child process exited after {} loops.", loops);

    Ok(())
}
