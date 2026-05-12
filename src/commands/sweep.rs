use crate::commands::load_library;
use crate::config::ResolvedConfig;
use crate::event::EventRegistry;
use crate::profiler::ProfilerBuilder;
use crate::scheduler::fixed::FixedScheduler;
use crate::sink::matrix::MatrixSink;
use crate::sink::perfetto::PerfettoSink;
use crate::sink::{self, OutputSink};
use crate::source::hardware::HardwareSampleSource;
use crate::state::propagate::PropagateEstimator;
use crate::syscalls;
use indicatif::{ProgressBar, ProgressStyle};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

pub fn sweep(
    library: Option<PathBuf>,
    config: ResolvedConfig,
    trace: PathBuf,
    matrix: Option<PathBuf>,
    quiet: bool,
    target: Vec<String>,
) -> std::io::Result<()> {
    let lib = load_library(library)?;
    let all_ids: Vec<u32> = (0..lib.events.len() as u32).collect();
    let batches: Vec<Vec<u32>> = all_ids.chunks(4).map(|c| c.to_vec()).collect();
    let num_batches = batches.len();
    tracing::info!(
        "Sweep: {} events across {} runs",
        all_ids.len(),
        num_batches
    );

    let registry = EventRegistry::new(lib.clone());
    let event_names: Vec<String> = all_ids
        .iter()
        .map(|&id| registry.get_event_name(id).to_string())
        .collect();

    let mut sinks: Vec<Box<dyn OutputSink>> = Vec::new();
    sinks.push(Box::new(PerfettoSink::new(
        &trace,
        event_names.clone(),
        0, // emit every quantum
    )?));
    if let Some(matrix_path) = matrix {
        sinks.push(Box::new(MatrixSink::new(
            matrix_path,
            event_names,
            config.q_sample_ns,
        )));
    }

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta_precise})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb
    };

    for (batch_idx, batch) in batches.iter().enumerate() {
        let registry = EventRegistry::new(lib.clone());
        let counter_names = batch
            .iter()
            .map(|&id| registry.get_event_name(id))
            .collect::<Vec<_>>()
            .join(", ");
        pb.set_message(counter_names);

        for s in &mut sinks {
            s.begin_batch(batch_idx as u32, batch);
        }

        let mut child = unsafe {
            std::process::Command::new(&target[0])
                .args(&target[1..])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .pre_exec(crate::syscalls::ptrace_traceme)
                .spawn()
                .expect("Failed to spawn child process")
        };

        let pid = child.id();
        syscalls::wait_for_exec(pid)?;

        let source = HardwareSampleSource::new(pid, registry, None, config.q_sample_ns)
            .expect("Failed to create hardware source");

        let mut profiler = ProfilerBuilder::new()
            .source(source)
            .scheduler(FixedScheduler::new(batch.clone()), batch.clone())
            .map_err(|e| std::io::Error::other(e.to_string()))?
            .estimator(PropagateEstimator::new())
            .sinks(&mut sinks)
            .build();

        syscalls::ptrace_detach(pid)?;

        let quantum_dur = Duration::from_nanos(config.q_schedule_ns);
        while child
            .try_wait()
            .expect("Failed to wait for child")
            .is_none()
        {
            profiler.step();
            thread::sleep(quantum_dur);
        }
        child.wait().unwrap();
        drop(profiler);

        pb.inc(1);
    }

    pb.finish_and_clear();
    sink::finish_sinks(&mut sinks);
    tracing::info!("Sweep complete. Trace written to {:?}", trace);

    Ok(())
}
