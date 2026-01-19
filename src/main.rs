use clap::Parser;
use saccade::cli::{Cli, Commands};
use saccade::event_library::EventLibrary;
use saccade::event_registry::EventRegistry;
use saccade::perf::Perf;
use std::fs::File;
use std::io::BufReader;

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
        Commands::Run { library, target } => {
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
            let _registry = EventRegistry::new(lib);
        }
    }

    Ok(())
}
