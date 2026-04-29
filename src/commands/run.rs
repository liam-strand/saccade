use crate::commands::{load_library, spawn_child};
use crate::config::ResolvedConfig;
use crate::event::EventRegistry;
use crate::profiler::ProfilerBuilder;
use crate::sink::csv::CsvSink;
use crate::sink::perfetto::PerfettoSink;
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

    let mut child = spawn_child(&target)?;
    let pid = child.id();
    syscalls::wait_for_exec(pid)?;

    let source = HardwareSampleSource::new(pid, registry, None, config.q_sample_ns)
        .expect("Failed to create hardware source");

    let mut builder = ProfilerBuilder::new()
        .source(source)
        .scheduler_boxed(config.build_scheduler(), all_ids)
        .estimator_boxed(config.build_estimator())
        .add_sink(PerfettoSink::new(trace, event_names, config.q_output_ns)?);

    if let Some(path) = csv {
        builder = builder.add_sink(CsvSink::new(path)?);
    }

    let mut profiler = builder.build();

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
    profiler.finish_sinks();
    debug!("Child process exited after {} loops.", loops);

    Ok(())
}
