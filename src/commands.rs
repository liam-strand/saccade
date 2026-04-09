mod generate;
mod run;
mod simulate;
mod sweep;

pub use generate::generate;
pub use run::run;
pub use simulate::simulate;
pub use sweep::sweep;

use crate::event::EventLibrary;
use crate::perf::Perf;
use crate::syscalls;
use std::fs::File;
use std::io::BufReader;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;
use tracing::debug;

pub(crate) fn load_library(path: Option<PathBuf>) -> std::io::Result<EventLibrary> {
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

pub(crate) fn spawn_child(target: &[String]) -> std::io::Result<std::process::Child> {
    unsafe {
        Command::new(&target[0])
            .args(&target[1..])
            .pre_exec(syscalls::ptrace_traceme)
            .spawn()
    }
}
