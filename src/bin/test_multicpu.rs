use perf_event::{Builder, events};
use std::time::{Duration, Instant};

fn main() {
    let mut counters = Vec::new();
    let num_cpus = 16;

    // Create counter on each CPU
    for cpu in 0..num_cpus {
        match Builder::new(events::Hardware::INSTRUCTIONS)
            .one_cpu(cpu)
            .any_pid()
            .build()
        {
            Ok(mut c) => {
                c.enable().expect("Failed to enable");
                counters.push((cpu, c));
            }
            Err(e) => {
                println!("Failed to create counter on CPU {}: {}", cpu, e);
            }
        }
    }

    println!("Started reading from {} CPUs", counters.len());

    let start = Instant::now();
    let run_time = Duration::from_secs(1);
    let sleep_time = Duration::from_millis(10); // 100 Hz

    let mut tick = 0;
    while start.elapsed() < run_time {
        print!("Tick {:03}: ", tick);
        for (cpu, c) in counters.iter_mut() {
            match c.read() {
                Ok(val) => print!("c{}:{} ", cpu, val),
                Err(_e) => print!("c{}:ERR ", cpu),
            }
        }
        println!();
        tick += 1;
        std::thread::sleep(sleep_time);
    }
}
