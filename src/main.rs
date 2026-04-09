use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use saccade::buffered_output::Logger;
use saccade::cli::{Cli, Commands};
use saccade::event_library::EventLibrary;
use saccade::event_registry::EventRegistry;
use saccade::hardware_backend::HardwareBackend;
use saccade::oculomotor::Oculomotor;
use saccade::perf::Perf;
use saccade::perfetto::{self, PerfettoWriter};
use saccade::scheduler::Scheduler;
use saccade::scheduler::fixed::FixedScheduler;
use saccade::scheduler::random::RandomScheduler;
use saccade::scheduler::round_robin::RoundRobinScheduler;
use saccade::syscalls;
use saccade::virtual_backend::{TimeVaryingRates, VirtualBackend};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::thread;
use std::time::Duration;
use tracing::debug;

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
            let lib = match library {
                Some(path) => {
                    debug!("Loading event library from {:?}", path);
                    let file = File::open(path)?;
                    let reader = BufReader::new(file);
                    serde_json::from_reader(reader)?
                }
                None => {
                    debug!("Generating event library on the fly...");
                    EventLibrary::from_bytes(&Perf::list()).unwrap()
                }
            };

            let registry = EventRegistry::new(lib);
            let num_events = registry.get_event_ids().len();
            let event_names: Vec<String> = registry
                .get_event_ids()
                .iter()
                .map(|&id| registry.get_event_name(id).to_string())
                .collect();
            debug!("Loaded {} events.", num_events);
            debug!("Target program args: {:?}", target);

            debug!("Parent process PID: {}", std::process::id());
            let mut child = unsafe {
                Command::new(target[0].clone())
                    .args(&target[1..])
                    .pre_exec(syscalls::ptrace_traceme)
                    .spawn()
                    .expect("Failed to spawn child process")
            };
            debug!("Child process spawned.");

            let pid = child.id();
            syscalls::wait_for_exec(pid)?;

            debug!("Oculomotor starting at {}", syscalls::gettid().unwrap());

            // let mut scheduler = RandomScheduler::default();
            // scheduler.init(registry.get_event_ids());
        
            let scheduler = FixedScheduler::new(vec![registry.lookup("ic_fw32").unwrap()]);

            let logger = Logger::new("saccade.csv", 256_000)?;
            let logger_tx = logger.clone_sender().expect("Failed to get logger sender");
            let backend = HardwareBackend::new(pid, registry, logger_tx)
                .expect("Failed to create hardware backend");

            let trace_writer = match trace {
                Some(path) => {
                    let mut writer = PerfettoWriter::new(path, event_names)?;
                    writer.register_tracks()?;
                    Some(writer)
                }
                None => None,
            };

            let mut oculomotor = Oculomotor::new(
                Box::new(backend),
                Box::new(scheduler),
                num_events,
                Some(logger),
                trace_writer,
            );

            debug!("Oculomotor is ready.");

            syscalls::ptrace_detach(pid)?;

            let mut quantum = Duration::from_nanos(quantum);
            let mut loops = 0;
            while child
                .try_wait()
                .expect("Failed to wait for child")
                .is_none()
            {
                if let Some(duration) = oculomotor.step() {
                    quantum = duration;
                }
                thread::sleep(quantum);
                loops += 1;
            }
            debug!("Child process exited after {} loops.", loops);

            child.wait().unwrap();
        }
        Commands::Sweep {
            library,
            quantum,
            trace,
            target,
        } => {
            let lib = match library {
                Some(path) => {
                    debug!("Loading event library from {:?}", path);
                    let file = File::open(path)?;
                    let reader = BufReader::new(file);
                    serde_json::from_reader(reader)?
                }
                None => {
                    debug!("Generating event library on the fly...");
                    EventLibrary::from_bytes(&Perf::list()).unwrap()
                }
            };

            let all_ids: Vec<u32> = (0..lib.events.len() as u32).collect();
            let batches: Vec<Vec<u32>> = all_ids.chunks(4).map(|c| c.to_vec()).collect();
            let num_batches = batches.len();
            tracing::info!(
                "Sweep: {} events across {} runs",
                all_ids.len(),
                num_batches
            );

            // Accumulate per-event rate time-series across all runs:
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

            for (_run_idx, batch) in batches.iter().enumerate() {
                let registry = EventRegistry::new(lib.clone());
                let counter_names = batch
                    .iter()
                    .map(|&id| registry.get_event_name(id))
                    .collect::<Vec<_>>()
                    .join(", ");
                pb.set_message(counter_names);
                let num_events = registry.get_event_ids().len();

                let mut child = unsafe {
                    Command::new(target[0].clone())
                        .args(&target[1..])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .pre_exec(syscalls::ptrace_traceme)
                        .spawn()
                        .expect("Failed to spawn child process")
                };

                let pid = child.id();
                syscalls::wait_for_exec(pid)?;

                let scheduler = FixedScheduler::new(batch.clone());

                // TODO: make logger_tx optional in HardwareBackend; use dummy channel for now
                let (dummy_tx, _dummy_rx) =
                    std::sync::mpsc::sync_channel::<saccade::counter_backend::SaccadeSample>(1);
                let backend = HardwareBackend::new(pid, registry, dummy_tx)
                    .expect("Failed to create hardware backend");

                let mut oculomotor = Oculomotor::new(
                    Box::new(backend),
                    Box::new(scheduler),
                    num_events,
                    None,
                    None,
                );

                syscalls::ptrace_detach(pid)?;

                let quantum_dur = Duration::from_nanos(quantum);
                while child
                    .try_wait()
                    .expect("Failed to wait for child")
                    .is_none()
                {
                    oculomotor.step();
                    let ts = oculomotor.last_step_ns();
                    let vcs = oculomotor.vcs();
                    for &id in batch.iter() {
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
            let reader = BufReader::new(file);
            let lib: EventLibrary = serde_json::from_reader(reader)?;

            let registry = EventRegistry::new(lib);
            let num_events = registry.get_event_ids().len();
            let sim_event_names: Vec<String> = registry
                .get_event_ids()
                .iter()
                .map(|&id| registry.get_event_name(id).to_string())
                .collect();
            debug!("Loaded {} events.", num_events);

            debug!("Loading rate time-series from {:?}", rates_trace);
            let timeseries = perfetto::read_rate_timeseries(&rates_trace)?;

            // Resolve event names to EventIds
            let mut series_map: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
            for (name, data) in timeseries.series {
                if let Some(id) = registry.lookup(&name) {
                    debug!("Rate series: {} (id={}) -> {} points", name, id, data.len());
                    series_map.insert(id, data);
                } else {
                    tracing::warn!("Unknown event in rates trace: {}", name);
                }
            }
            let tv_rates = TimeVaryingRates { series: series_map };

            let mut scheduler: Box<dyn Scheduler> = match scheduler_name.as_str() {
                "random" => Box::new(RandomScheduler::default()),
                "round_robin" => Box::new(RoundRobinScheduler::default()),
                other => {
                    eprintln!("Unknown scheduler: {}. Using random.", other);
                    Box::new(RandomScheduler::default())
                }
            };
            scheduler.init(registry.get_event_ids());

            let logger = match output {
                Some(path) => Some(Logger::new(path, 256_000)?),
                None => None,
            };
            let logger_tx = logger.as_ref().and_then(|l| l.clone_sender());

            let backend = VirtualBackend::new(tv_rates, 0.0, quantum, None, logger_tx);

            let trace_writer = match trace {
                Some(path) => {
                    let mut writer = PerfettoWriter::new(path, sim_event_names)?;
                    writer.register_tracks()?;
                    Some(writer)
                }
                None => None,
            };

            let mut oculomotor = Oculomotor::new(
                Box::new(backend),
                scheduler,
                num_events,
                logger,
                trace_writer,
            );

            tracing::info!("Simulating {} steps (quantum={}ns)...", steps, quantum);

            for _ in 0..steps {
                oculomotor.step();
            }

            // Print VCS summary
            let vcs = oculomotor.vcs();
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
