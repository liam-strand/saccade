use perf_event::{Builder, events};
use std::time::Duration;

fn main() {
    let mut counters = vec![];

    // Initial 4
    for _ in 0..4 {
        let mut c = Builder::new(events::Hardware::CPU_CYCLES)
            .one_cpu(0)
            .any_pid()
            .build()
            .unwrap();
        c.enable().unwrap();
        counters.push(c);
    }

    println!("Initial setup:");
    for (i, c) in counters.iter_mut().enumerate() {
        println!("Counter {}: {}", i, c.read().unwrap());
    }

    // Now, dynamic rotation loop mimicking oculomotor
    for loop_idx in 0..10 {
        std::thread::sleep(Duration::from_millis(50));

        let slot = loop_idx % 4; // rotate one slot
        counters.remove(slot); // drop old HW counter

        let mut new_counter = Builder::new(events::Hardware::INSTRUCTIONS)
            .one_cpu(0)
            .any_pid()
            .build()
            .unwrap();
        new_counter.enable().unwrap();
        counters.insert(slot, new_counter); // assign new

        println!("After rotation {}:", loop_idx);
        for (i, c) in counters.iter_mut().enumerate() {
            println!("Counter {}: {}", i, c.read().unwrap());
        }
    }
}
