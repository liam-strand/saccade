use crate::commands::load_library;
use crate::config::ResolvedConfig;
use crate::event::EventRegistry;
use crate::perfetto::PerfettoWriter;
use crate::sample::TASK_COMM_LEN;
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
    config: ResolvedConfig,
    trace: PathBuf,
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

    // (event_id, synthetic_tid) -> Vec<(timestamp_ns, rate)>
    let mut all_series: HashMap<(u32, u32), Vec<(u64, f64)>> = HashMap::new();
    // synthetic_tid -> (tgid, task_name)
    let mut thread_meta: HashMap<u32, (u32, String)> = HashMap::new();
    // Global across batches: (task_name, within_name_idx) -> synthetic_tid
    let mut name_idx_to_synthetic: HashMap<(String, u32), u32> = HashMap::new();
    let mut next_synthetic_tid: u32 = 1;

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

        let mut source = HardwareSampleSource::new(pid, registry, None, config.q_sample_ns)
            .expect("Failed to create hardware source");

        source
            .apply_schedule(&[], batch)
            .expect("Failed to apply schedule");
        syscalls::ptrace_detach(pid)?;

        let mut batch_real_to_synthetic: HashMap<u32, u32> = HashMap::new();
        let mut batch_name_counters: HashMap<String, u32> = HashMap::new();

        let quantum_dur = Duration::from_nanos(config.q_schedule_ns);
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

                let task_len = s.task.iter().position(|&c| c == 0).unwrap_or(TASK_COMM_LEN);
                let task_name = String::from_utf8_lossy(&s.task[..task_len]).into_owned();
                let synthetic_tid = *batch_real_to_synthetic.entry(s.tid).or_insert_with(|| {
                    let counter = batch_name_counters.entry(task_name.clone()).or_insert(0);
                    let within_name_idx = *counter;
                    *counter += 1;
                    *name_idx_to_synthetic
                        .entry((task_name.clone(), within_name_idx))
                        .or_insert_with(|| {
                            let id = next_synthetic_tid;
                            next_synthetic_tid += 1;
                            id
                        })
                });

                all_series
                    .entry((s.event_id, synthetic_tid))
                    .or_default()
                    .push((rel_ts, s.count as f64 / s.duration_ns as f64));
                thread_meta
                    .entry(synthetic_tid)
                    .or_insert((0u32, task_name));
            }
            thread::sleep(quantum_dur);
        }
        child.wait().unwrap();
        pb.inc(1);
    }
    pb.finish_and_clear();

    let registry = EventRegistry::new(lib.clone());
    let event_names: Vec<String> = (0..lib.events.len() as u32)
        .map(|id| registry.get_event_name(id).to_string())
        .collect();
    let mut writer = PerfettoWriter::new(&trace, event_names)?;
    writer.write_raw_series(&all_series, &thread_meta)?;
    writer.flush()?;
    tracing::info!("Sweep complete. Trace written to {:?}", trace);

    Ok(())
}
