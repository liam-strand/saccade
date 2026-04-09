use crate::commands::{load_library, spawn_child};
use crate::event::EventRegistry;
use crate::profiler::ProfilerBuilder;
use crate::scheduler::fixed::FixedScheduler;
use crate::sink::csv::CsvSink;
use crate::sink::perfetto::PerfettoSink;
use crate::source::SampleSource;
use crate::source::hardware::HardwareSampleSource;
use crate::syscalls;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tracing::debug;

pub fn run(
    library: Option<PathBuf>,
    quantum: u64,
    trace: Option<PathBuf>,
    target: Vec<String>,
) -> std::io::Result<()> {
    let lib = load_library(library)?;
    let registry = EventRegistry::new(lib);
    let num_events = registry.get_event_ids().len();
    let all_ids = registry.get_event_ids();
    let event_names: Vec<String> = all_ids
        .iter()
        .map(|&id| registry.get_event_name(id).to_string())
        .collect();
    debug!("Loaded {} events.", num_events);

    let mut child = spawn_child(&target)?;
    let pid = child.id();
    syscalls::wait_for_exec(pid)?;

    let source =
        HardwareSampleSource::new(pid, registry, None).expect("Failed to create hardware source");

    // TODO: make the initial event set configurable via --events flag
    let initial_events: Vec<u32> = all_ids[..source.num_slots().min(all_ids.len())].to_vec();
    let scheduler = FixedScheduler::new(initial_events);

    let mut builder = ProfilerBuilder::new()
        .num_events(num_events)
        .source(source)
        .scheduler(scheduler, all_ids)
        .add_sink(CsvSink::new("saccade.csv")?);

    if let Some(path) = trace {
        builder = builder.add_sink(PerfettoSink::new(path, event_names)?);
    }

    let mut profiler = builder.build();

    debug!("Profiler is ready.");
    syscalls::ptrace_detach(pid)?;

    let mut quantum_dur = Duration::from_nanos(quantum);
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
