use clap::Parser;
use saccade::cli::{Cli, Commands};
use saccade::event_library::EventLibrary;
use saccade::event_registry::EventRegistry;
use saccade::oculomotor::Oculomotor;
use saccade::perf::Perf;
use saccade::scheduler::Scheduler;
use saccade::scheduler::random::RandomScheduler;
use saccade::syscalls;
use std::fs::File;
use std::io::BufReader;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { output } => {
            println!("Generating event library to {:?}", output);
            let lib = EventLibrary::from_bytes(&Perf::list()).unwrap();
            let buf = File::create(output)?;
            serde_json::to_writer_pretty(buf, &lib)?;
            println!("Successfully generated event library.");
        }
        Commands::Run {
            library,
            quantum,
            target,
        } => {
            let lib = match library {
                Some(path) => {
                    println!("Loading event library from {:?}", path);
                    let file = File::open(path)?;
                    let reader = BufReader::new(file);
                    serde_json::from_reader(reader)?
                }
                None => {
                    println!("Generating event library on the fly...");
                    EventLibrary::from_bytes(&Perf::list()).unwrap()
                }
            };
            println!("Loaded {} events.", lib.events.len());
            println!("Target program args: {:?}", target);

            let (tx, rx) = channel();
            // Ready channel to synchronize Oculomotor initialization
            let (ready_tx, ready_rx) = channel();

            eprintln!("Parent process PID: {}", std::process::id());
            let mut child = unsafe {
                Command::new(target[0].clone())
                    .args(&target[1..])
                    .pre_exec(|| syscalls::ptrace_traceme())
                    .spawn()
                    .expect("Failed to spawn child process")
            };
            eprintln!("Child process spawned.");

            let pid = child.id();

            // Wait for child to stop at exec
            syscalls::wait_for_exec(pid)?;

            let pid = child.id();
            let thread = thread::spawn(move || {
                let registry = EventRegistry::new(lib);
                let mut scheduler = RandomScheduler::default();
                scheduler.init(registry.get_event_ids());
                let mut oculomotor = Oculomotor::new(
                    pid,
                    registry,
                    Box::new(scheduler),
                    std::path::PathBuf::from("saccade.csv"),
                )
                .unwrap();

                ready_tx.send(()).unwrap();

                let mut done = false;
                let mut quantum = Duration::from_nanos(quantum);
                let mut loops = 0;
                while !done {
                    if let Some(duration) = oculomotor.step() {
                        quantum = duration;
                    }
                    if rx.try_recv().is_ok() {
                        done = true;
                    }
                    thread::sleep(quantum);
                    loops += 1;
                }
                println!("Oculomotor looped {} times.", loops);
            });

            eprintln!("Oculomotor thread spawned.");

            // Wait for Oculomotor to be ready
            ready_rx
                .recv()
                .expect("Failed to receive ready signal from Oculomotor");

            eprintln!("Oculomotor is ready.");

            // Resume the child process: PTRACE_DETACH
            syscalls::ptrace_detach(pid)?;

            eprintln!("Child process resumed.");

            child.wait().unwrap();
            tx.send(()).unwrap();
            thread.join().unwrap();

            eprintln!("Oculomotor thread joined.");
        }
    }

    Ok(())
}
