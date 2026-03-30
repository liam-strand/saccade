use clap::Parser;
use saccade::buffered_output::Logger;
use saccade::cli::{Cli, Commands};
use saccade::event_library::EventLibrary;
use saccade::event_registry::EventRegistry;
use saccade::hardware_backend::HardwareBackend;
use saccade::oculomotor::Oculomotor;
use saccade::perf::Perf;
use saccade::scheduler::Scheduler;
use saccade::scheduler::random::RandomScheduler;
use saccade::scheduler::round_robin::RoundRobinScheduler;
use saccade::syscalls;
use saccade::virtual_backend::{GoldenRates, VirtualBackend};
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

            let mut scheduler = RandomScheduler::default();
            scheduler.init(registry.get_event_ids());

            let logger = Logger::new("saccade.csv", 256_000)?;
            let logger_tx = logger.clone_sender().expect("Failed to get logger sender");
            let backend = HardwareBackend::new(pid, registry, logger_tx)
                .expect("Failed to create hardware backend");

            let mut oculomotor = Oculomotor::new(
                Box::new(backend),
                Box::new(scheduler),
                num_events,
                Some(logger),
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
        Commands::Simulate {
            library,
            golden,
            quantum,
            steps,
            output,
            scheduler: scheduler_name,
        } => {
            debug!("Loading event library from {:?}", library);
            let file = File::open(library)?;
            let reader = BufReader::new(file);
            let lib: EventLibrary = serde_json::from_reader(reader)?;

            let registry = EventRegistry::new(lib);
            let num_events = registry.get_event_ids().len();
            debug!("Loaded {} events.", num_events);

            debug!("Loading golden rates from {:?}", golden);
            let golden_file = File::open(golden)?;
            let golden_reader = BufReader::new(golden_file);
            let golden_rates: GoldenRates =
                serde_json::from_reader(golden_reader).expect("Failed to parse golden rates JSON");

            // Resolve event names to EventIds
            let mut rate_map: HashMap<u32, f64> = HashMap::new();
            for (name, rate) in &golden_rates.rates {
                if let Some(id) = registry.lookup(name) {
                    rate_map.insert(id, *rate);
                    debug!("Golden rate: {} (id={}) -> {} events/ns", name, id, rate);
                } else {
                    tracing::warn!("Unknown event in golden rates: {}", name);
                }
            }

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

            let backend = VirtualBackend::new(
                rate_map,
                golden_rates.noise_stddev,
                quantum,
                golden_rates.seed,
                logger_tx,
            );

            let mut oculomotor = Oculomotor::new(Box::new(backend), scheduler, num_events, logger);

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
