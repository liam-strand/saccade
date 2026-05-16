//! Thin wrapper around the `perf` CLI for querying available hardware performance counter events.

use std::process::Command;

/// Zero-sized namespace struct grouping `perf` CLI invocations.
pub struct Perf {}

impl Perf {
    /// Runs `perf list --details` and returns its raw stdout bytes for parsing by [`crate::event::EventLibrary`].
    pub fn list() -> Vec<u8> {
        let output = Command::new("perf")
            .arg("list")
            .arg("--details")
            .output()
            .expect("Failed to run perf");
        output.stdout
    }
}
