use crate::event::EventLibrary;
use crate::perf::Perf;
use std::fs::File;
use std::path::PathBuf;

pub fn generate(output: PathBuf) -> std::io::Result<()> {
    tracing::info!("Generating event library to {:?}", output);
    let lib = EventLibrary::from_bytes(&Perf::list()).unwrap();
    let buf = File::create(output)?;
    serde_json::to_writer_pretty(buf, &lib)?;
    tracing::info!("Successfully generated event library.");
    Ok(())
}
