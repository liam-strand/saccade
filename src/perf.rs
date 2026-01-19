use std::process::Command;

pub struct Perf {}

impl Perf {
    pub fn list() -> Vec<u8> {
        let output = Command::new("perf")
            .arg("list")
            .arg("--details")
            .output()
            .expect("Failed to run perf");
        output.stdout
    }
}
