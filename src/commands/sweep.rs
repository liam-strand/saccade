use crate::commands::load_library;
use crate::event::EventRegistry;
use crate::perfetto::PerfettoWriter;
use crate::source::SampleSource;
use crate::source::hardware::HardwareSampleSource;
use crate::syscalls;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

pub fn sweep(
    library: Option<PathBuf>,
    quantum: u64,
    trace: Option<PathBuf>,
    target: Vec<String>,
) -> std::io::Result<()> {
    let lib = load_library(library)?;
    let all_ids: Vec<u32> = (0..lib.events.len() as u32).collect();
    let batches: Vec<Vec<u32>> = all_ids.chunks(4).map(|c| c.to_vec()).collect();
    let num_batches = batches.len();
    tracing::info!(
        "Sweep: {} events across {} runs",
        all_ids.len(),
        num_batches
    );

    // event_id -> Vec<(timestamp_ns, rate)>
    let mut all_series: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();

    let pb = ProgressBar::new(num_batches as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta_precise})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    for batch in &batches {
        let registry = EventRegistry::new(lib.clone());
        let counter_names = batch
            .iter()
            .map(|&id| registry.get_event_name(id))
            .collect::<Vec<_>>()
            .join(", ");
        pb.set_message(counter_names);

        let mut child = unsafe {
            std::process::Command::new(&target[0])
                .args(&target[1..])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .pre_exec(crate::syscalls::ptrace_traceme)
                .spawn()
                .expect("Failed to spawn child process")
        };

        let pid = child.id();
        syscalls::wait_for_exec(pid)?;

        let mut source = HardwareSampleSource::new(pid, registry, None)
            .expect("Failed to create hardware source");

        source
            .apply_schedule(&[], batch)
            .expect("Failed to apply schedule");
        syscalls::ptrace_detach(pid)?;

        let quantum_dur = Duration::from_nanos(quantum);
        let mut batch_t0: Option<u64> = None;
        while child
            .try_wait()
            .expect("Failed to wait for child")
            .is_none()
        {
            let (raw_samples, _elapsed_ns) = source.collect();
            for s in raw_samples {
                assert_ne!(s.duration_ns, 0);
                let t0 = *batch_t0.get_or_insert(s.timestamp_ns);
                let rel_ts = s.timestamp_ns.saturating_sub(t0);
                all_series
                    .entry(s.event_id)
                    .or_default()
                    .push((rel_ts, s.count as f64 / s.duration_ns as f64));
            }
            thread::sleep(quantum_dur);
        }
        child.wait().unwrap();
        pb.inc(1);
    }
    pb.finish_and_clear();

    match trace {
        Some(path) => {
            let registry = EventRegistry::new(lib.clone());
            let event_names: Vec<String> = (0..lib.events.len() as u32)
                .map(|id| registry.get_event_name(id).to_string())
                .collect();
            let mut writer = PerfettoWriter::new(&path, event_names)?;
            writer.register_tracks()?;
            writer.write_raw_series(&all_series)?;
            writer.flush()?;
            tracing::info!("Sweep complete. Trace written to {:?}", path);
        }
        None => {
            tracing::info!("Sweep complete. (No --trace specified; results discarded.)");
        }
    }

    Ok(())
}
