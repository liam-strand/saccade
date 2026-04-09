use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use saccade::cli::{Cli, Commands};
use saccade::event_library::EventLibrary;
use saccade::event_registry::EventRegistry;
use saccade::perf::Perf;
use saccade::perfetto::{self, PerfettoWriter};
use saccade::profiler::ProfilerBuilder;
use saccade::scheduler::fixed::FixedScheduler;
use saccade::scheduler::random::RandomScheduler;
use saccade::scheduler::round_robin::RoundRobinScheduler;
use saccade::sink::csv::CsvSink;
use saccade::sink::null::NullSink;
use saccade::sink::perfetto::PerfettoSink;
use saccade::source::SampleSource;
use saccade::source::hardware::HardwareSampleSource;
use saccade::source::virtual_source::VirtualSampleSource;
use saccade::syscalls;
use saccade::virtual_backend::TimeVaryingRates;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::thread;
use std::time::Duration;
use tracing::debug;

fn load_library(path: Option<std::path::PathBuf>) -> std::io::Result<EventLibrary> {
    match path {
        Some(p) => {
            debug!("Loading event library from {:?}", p);
            let file = File::open(p)?;
            let reader = BufReader::new(file);
            Ok(serde_json::from_reader(reader)?)
        }
        None => {
            debug!("Generating event library on the fly...");
            Ok(EventLibrary::from_bytes(&Perf::list()).unwrap())
        }
    }
}

fn spawn_child(target: &[String]) -> std::io::Result<std::process::Child> {
    unsafe {
        Command::new(&target[0])
            .args(&target[1..])
            .pre_exec(syscalls::ptrace_traceme)
            .spawn()
    }
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    let level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    match cli.command {
        Commands::Generate { output } => {
            tracing::info!("Generating event library to {:?}", output);
            let lib = EventLibrary::from_bytes(&Perf::list()).unwrap();
            let buf = File::create(output)?;
            serde_json::to_writer_pretty(buf, &lib)?;
            tracing::info!("Successfully generated event library.");
        }

        Commands::Run {
            library,
            quantum,
            trace,
            target,
        } => {
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

            let source = HardwareSampleSource::new(pid, registry, None)
                .expect("Failed to create hardware source");

            // TODO: make the initial event set configurable via --events flag
            let initial_events: Vec<u32> =
                all_ids[..source.num_slots().min(all_ids.len())].to_vec();
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
            while child.try_wait().expect("Failed to wait for child").is_none() {
                if let Some(d) = profiler.step() {
                    quantum_dur = d;
                }
                thread::sleep(quantum_dur);
                loops += 1;
            }
            child.wait().unwrap();
            profiler.finish_sinks();
            debug!("Child process exited after {} loops.", loops);
        }

        Commands::Sweep {
            library,
            quantum,
            trace,
            target,
        } => {
            let lib = load_library(library)?;
            let all_ids: Vec<u32> = (0..lib.events.len() as u32).collect();
            let batches: Vec<Vec<u32>> = all_ids.chunks(4).map(|c| c.to_vec()).collect();
            let num_batches = batches.len();
            tracing::info!(
                "Sweep: {} events across {} runs",
                all_ids.len(),
                num_batches
            );

            // event_id -> Vec<(timestamp_ns, rate)>
            let mut all_series: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();

            let pb = ProgressBar::new(num_batches as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta_precise})",
                )
                .unwrap()
                .progress_chars("=>-"),
            );

            for batch in &batches {
                let registry = EventRegistry::new(lib.clone());
                let counter_names = batch
                    .iter()
                    .map(|&id| registry.get_event_name(id))
                    .collect::<Vec<_>>()
                    .join(", ");
                pb.set_message(counter_names);
                let num_events = registry.get_event_ids().len();

                let mut child = unsafe {
                    Command::new(&target[0])
                        .args(&target[1..])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .pre_exec(syscalls::ptrace_traceme)
                        .spawn()
                        .expect("Failed to spawn child process")
                };

                let pid = child.id();
                syscalls::wait_for_exec(pid)?;

                let source =
                    HardwareSampleSource::new(pid, registry, None)
                        .expect("Failed to create hardware source");

                let scheduler = FixedScheduler::new(batch.clone());

                let mut profiler = ProfilerBuilder::new()
                    .num_events(num_events)
                    .source(source)
                    .scheduler(scheduler, (0..num_events as u32).collect())
                    .add_sink(NullSink)
                    .build();

                syscalls::ptrace_detach(pid)?;

                let quantum_dur = Duration::from_nanos(quantum);
                while child.try_wait().expect("Failed to wait for child").is_none() {
                    profiler.step();
                    let ts = profiler.current_time_ns();
                    let vcs = profiler.vcs();
                    for &id in batch {
                        let est = &vcs.all_estimates()[id as usize];
                        if est.sample_count > 0 {
                            all_series.entry(id).or_default().push((ts, est.rate));
                        }
                    }
                    thread::sleep(quantum_dur);
                }
                child.wait().unwrap();
                pb.inc(1);
            }
            pb.finish_and_clear();

            match trace {
                Some(path) => {
                    let registry = EventRegistry::new(lib.clone());
                    let event_names: Vec<String> = (0..lib.events.len() as u32)
                        .map(|id| registry.get_event_name(id).to_string())
                        .collect();
                    let mut writer = PerfettoWriter::new(&path, event_names)?;
                    writer.register_tracks()?;
                    writer.write_raw_series(&all_series)?;
                    writer.flush()?;
                    tracing::info!("Sweep complete. Trace written to {:?}", path);
                }
                None => {
                    tracing::info!("Sweep complete. (No --trace specified; results discarded.)");
                }
            }
        }

        Commands::Simulate {
            library,
            rates_trace,
            quantum,
            steps,
            output,
            scheduler: scheduler_name,
            trace,
        } => {
            debug!("Loading event library from {:?}", library);
            let file = File::open(library)?;
            let lib: EventLibrary = serde_json::from_reader(BufReader::new(file))?;

            let registry = EventRegistry::new(lib);
            let num_events = registry.get_event_ids().len();
            let all_ids = registry.get_event_ids();
            let event_names: Vec<String> = all_ids
                .iter()
                .map(|&id| registry.get_event_name(id).to_string())
                .collect();
            debug!("Loaded {} events.", num_events);

            debug!("Loading rate time-series from {:?}", rates_trace);
            let timeseries = perfetto::read_rate_timeseries(&rates_trace)?;

            let mut series_map: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
            for (name, data) in timeseries.series {
                if let Some(id) = registry.lookup(&name) {
                    debug!("Rate series: {} (id={}) -> {} points", name, id, data.len());
                    series_map.insert(id, data);
                } else {
                    tracing::warn!("Unknown event in rates trace: {}", name);
                }
            }

            let source =
                VirtualSampleSource::new(TimeVaryingRates { series: series_map }, 0.0, quantum, None, 4);

            let mut builder = ProfilerBuilder::new()
                .num_events(num_events)
                .source(source);

            builder = match scheduler_name.as_str() {
                "random" => builder.scheduler(RandomScheduler::default(), all_ids),
                "round_robin" => builder.scheduler(RoundRobinScheduler::default(), all_ids),
                other => {
                    eprintln!("Unknown scheduler: {}. Using random.", other);
                    builder.scheduler(RandomScheduler::default(), all_ids)
                }
            };

            if let Some(path) = output {
                builder = builder.add_sink(CsvSink::new(path)?);
            }
            if let Some(path) = trace {
                builder = builder.add_sink(PerfettoSink::new(path, event_names)?);
            }

            let mut profiler = builder.build();

            tracing::info!("Simulating {} steps (quantum={}ns)...", steps, quantum);
            for _ in 0..steps {
                profiler.step();
            }
            profiler.finish_sinks();

            // Print VCS summary
            let vcs = profiler.vcs();
            eprintln!(
                "\n{:<6} {:<14} {:<14} Samples",
                "ID", "Rate (ev/ns)", "Uncertainty"
            );
            eprintln!("{}", "-".repeat(50));
            for (i, est) in vcs.all_estimates().iter().enumerate() {
                if est.sample_count > 0 || est.rate > 0.0 {
                    eprintln!(
                        "{:<6} {:<14.6} {:<14.6} {}",
                        i, est.rate, est.uncertainty, est.sample_count
                    );
                }
            }
            tracing::info!("Simulation complete.");
        }
    }

    Ok(())
}
