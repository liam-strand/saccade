use clap::Parser;
use saccade::cli::{Cli, Commands};
use saccade::event_library::EventLibrary;
use saccade::event_registry::EventRegistry;
use saccade::oculomotor::Oculomotor;
use saccade::perf::Perf;
use saccade::scheduler::round_robin::RoundRobinScheduler;
use std::fs::File;
use std::io::BufReader;
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
            let thread = thread::spawn(move || {
                let registry = EventRegistry::new(lib);
                let scheduler = RoundRobinScheduler::default();
                let mut oculomotor = Oculomotor::new(0, registry, Box::new(scheduler)).unwrap();
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

            let mut child = Command::new(target[0].clone())
                .args(&target[1..])
                .spawn()
                .expect("Failed to spawn child process");

            child.wait().unwrap();
            tx.send(()).unwrap();
            thread.join().unwrap();
        }
    }

    Ok(())
}
