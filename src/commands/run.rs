use crate::commands::{load_library, spawn_child};
use crate::event::EventRegistry;
use crate::profiler::ProfilerBuilder;
use crate::scheduler::Scheduler;
use crate::scheduler::round_robin::RoundRobinScheduler;
use crate::sink::csv::CsvSink;
use crate::sink::perfetto::PerfettoSink;
use crate::source::SampleSource;
use crate::source::hardware::HardwareSampleSource;
use crate::state::propagate::PropagateEstimator;
use crate::syscalls;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tracing::debug;

pub fn run(
    library: Option<PathBuf>,
    q_schedule: u64,
    q_sample: u64,
    q_output: u64,
    trace: Option<PathBuf>,
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

    let source = HardwareSampleSource::new(pid, registry, None, q_sample)
        .expect("Failed to create hardware source");

    let mut scheduler = RoundRobinScheduler::new();
    scheduler.init(all_ids.clone(), source.num_slots());

    let mut builder = ProfilerBuilder::new()
        .source(source)
        .scheduler(scheduler, all_ids)
        .estimator(PropagateEstimator::new())
        .add_sink(CsvSink::new("saccade.csv")?);

    if let Some(path) = trace {
        builder = builder.add_sink(PerfettoSink::new(path, event_names, q_output)?);
    }

    let mut profiler = builder.build();

    debug!("Profiler is ready.");
    syscalls::ptrace_detach(pid)?;

    let mut quantum_dur = Duration::from_nanos(q_schedule);
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
