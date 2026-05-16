//! Implementation of the `generate` subcommand: build and persist a hardware-event library by querying `perf list`.

use crate::event::EventLibrary;
use crate::perf::Perf;
use std::fs::File;
use std::path::PathBuf;

/// Queries `perf list`, parses the output into an `EventLibrary`, and serialises it as pretty JSON to `output`.
pub fn generate(output: PathBuf) -> std::io::Result<()> {
    tracing::info!("Generating event library to {:?}", output);
    let lib = EventLibrary::from_bytes(&Perf::list()).unwrap();
    let buf = File::create(output)?;
    serde_json::to_writer_pretty(buf, &lib)?;
    tracing::info!("Successfully generated event library.");
    Ok(())
}
