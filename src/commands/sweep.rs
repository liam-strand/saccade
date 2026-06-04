//! Implementation of the `sweep` subcommand: exhaustively measure every hardware event by cycling
//! the target through fixed-counter batches.
//!
//! Each batch reserves one counter slot for `ex_ret_instr` (retired instructions) as an anchor and
//! fills the remaining slots with user events.  After all batches complete, every non-anchor
//! event rate is normalized to `(count / anchor_count) * global_ref_rate` so that rates are
//! comparable across batches regardless of run-to-run timing variation.  The anchor event's own
//! track shows the instruction throughput (events/ns) over time.

use crate::commands::load_library;
use crate::config::ResolvedConfig;
use crate::event::EventId;
use crate::event::EventRegistry;
use crate::perfetto::{PerfettoWriter, read_rate_timeseries};
use crate::profiler::ProfilerBuilder;
use crate::quantum::Quantum;
use crate::sample::{MAX_COUNTERS, TASK_COMM_LEN};
use crate::scheduler::fixed::FixedScheduler;
use crate::sink::matrix::MatrixSink;
use crate::sink::perfetto::PerfettoSink;
use crate::sink::{self, OutputSink};
use crate::source::hardware::HardwareSampleSource;
use crate::state::StateEstimator;
use crate::state::propagate::PropagateEstimator;
use crate::syscalls;
use indicatif::{ProgressBar, ProgressStyle};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::thread;
use std::time::Duration;

/// Maps real TID → (tgid, task_name), one HashMap per batch.
type BatchMaps = Vec<HashMap<u32, (u32, String)>>;

/// Accumulates the anchor event's raw count and duration across all quanta and batches.
/// Used after the sweep loop to compute the global instruction rate for normalization.
/// Shares accumulated totals with the outer scope via `Rc<RefCell<(total_count, total_duration_ns)>>`.
struct InstructionTrackerSink {
    anchor_id: EventId,
    totals: Rc<RefCell<(u64, u64)>>,
}

impl OutputSink for InstructionTrackerSink {
    fn emit(
        &mut self,
        quantum: &Quantum,
        _estimator: &dyn StateEstimator,
        _active_set: &[EventId],
    ) -> io::Result<()> {
        if let Some(agg) = quantum.aggregates().get(&self.anchor_id) {
            let mut t = self.totals.borrow_mut();
            t.0 += agg.total_count;
            t.1 += agg.total_duration_ns;
        }
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Lightweight sink that records, for each batch, which real TIDs appeared and what task they belong to.
struct TidCollectorSink {
    maps: Rc<RefCell<BatchMaps>>,
}

impl OutputSink for TidCollectorSink {
    fn begin_batch(&mut self, _batch_id: u32, _events: &[EventId]) {
        self.maps.borrow_mut().push(HashMap::new());
    }

    fn emit(
        &mut self,
        quantum: &Quantum,
        _estimator: &dyn StateEstimator,
        _active_set: &[EventId],
    ) -> io::Result<()> {
        let mut maps = self.maps.borrow_mut();
        let map = maps
            .last_mut()
            .expect("begin_batch must be called before emit");
        for s in quantum.samples() {
            if s.tid == 0 {
                continue;
            }
            map.entry(s.tid).or_insert_with(|| {
                let task_len = s.task.iter().position(|&c| c == 0).unwrap_or(TASK_COMM_LEN);
                (
                    s.pid,
                    String::from_utf8_lossy(&s.task[..task_len]).into_owned(),
                )
            });
        }
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Read the Perfetto trace at `trace`, assign synthetic TIDs (sorted-TID order within each
/// task_name group, reproducible across runs), normalize non-anchor rates to instruction-count,
/// and overwrite the trace.
fn remap_sweep_tids(
    trace: &Path,
    batch_maps: &[HashMap<u32, (u32, String)>],
    event_names: &[String],
    anchor_event_name: &str,
    global_ref_rate: f64,
) -> io::Result<()> {
    let ts = read_rate_timeseries(trace)?;

    // Pre-build per-real_tid anchor rate maps. Each batch run produces its own real_tid
    // for the target process, so anchor data keyed by real_tid is batch-specific.
    let anchor_maps: HashMap<u32, HashMap<u64, f64>> = ts
        .series
        .iter()
        .filter(|((name, _), _)| name == anchor_event_name)
        .map(|((_, tid), pts)| (*tid, pts.iter().copied().collect()))
        .collect();

    // Build real_tid → batch_index.
    let mut tid_to_batch: HashMap<u32, usize> = HashMap::new();
    for (batch_idx, map) in batch_maps.iter().enumerate() {
        for &real_tid in map.keys() {
            tid_to_batch.entry(real_tid).or_insert(batch_idx);
        }
    }

    // Assign synthetic TIDs: within each batch, group real TIDs by task_name, sort each group
    // by real TID value, then assign ordinals. Same (task_name, ordinal) → same synthetic TID
    // across batches, making evaluate() comparisons work across sweep runs.
    let mut name_idx_to_synthetic: HashMap<(String, usize), u32> = HashMap::new();
    let mut next_synthetic: u32 = 1;
    let mut batch_real_to_synthetic: Vec<HashMap<u32, u32>> =
        vec![HashMap::new(); batch_maps.len()];

    for (batch_idx, map) in batch_maps.iter().enumerate() {
        let mut by_name: HashMap<&str, Vec<u32>> = HashMap::new();
        for (&real_tid, (_, task_name)) in map {
            by_name
                .entry(task_name.as_str())
                .or_default()
                .push(real_tid);
        }
        for tids in by_name.values_mut() {
            tids.sort_unstable();
        }
        for (task_name, tids) in &by_name {
            for (ordinal, &real_tid) in tids.iter().enumerate() {
                let key = (task_name.to_string(), ordinal);
                let syn = *name_idx_to_synthetic.entry(key).or_insert_with(|| {
                    let id = next_synthetic;
                    next_synthetic += 1;
                    id
                });
                batch_real_to_synthetic[batch_idx].insert(real_tid, syn);
            }
        }
    }

    // Build event_name → event_id index.
    let event_name_to_id: HashMap<&str, EventId> = event_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i as EventId))
        .collect();

    // Remap (event_name, real_tid) → (event_id, synthetic_tid), normalizing non-anchor rates.
    let mut remapped: HashMap<(EventId, u32), Vec<(u64, f64)>> = HashMap::new();
    for ((event_name, real_tid), pts) in &ts.series {
        let Some(&event_id) = event_name_to_id.get(event_name.as_str()) else {
            continue;
        };
        let Some(&batch_idx) = tid_to_batch.get(real_tid) else {
            continue;
        };
        let Some(&syn_tid) = batch_real_to_synthetic[batch_idx].get(real_tid) else {
            continue;
        };
        let normalized_pts: Vec<(u64, f64)> = if event_name == anchor_event_name {
            pts.clone()
        } else {
            pts.iter()
                .map(|&(ts_ns, rate)| {
                    let anchor_rate = anchor_maps
                        .get(real_tid)
                        .and_then(|m| m.get(&ts_ns))
                        .copied()
                        .unwrap_or(0.0);
                    let norm = if anchor_rate > 0.0 {
                        (rate / anchor_rate) * global_ref_rate
                    } else {
                        rate
                    };
                    (ts_ns, norm)
                })
                .collect()
        };
        remapped
            .entry((event_id, syn_tid))
            .or_default()
            .extend_from_slice(&normalized_pts);
    }

    // Build thread_meta: synthetic_tid → (tgid, task_name).
    let mut thread_meta: HashMap<u32, (u32, String)> = HashMap::new();
    for (batch_idx, map) in batch_maps.iter().enumerate() {
        for (&real_tid, (tgid, task_name)) in map {
            if let Some(&syn_tid) = batch_real_to_synthetic[batch_idx].get(&real_tid) {
                thread_meta
                    .entry(syn_tid)
                    .or_insert_with(|| (*tgid, task_name.clone()));
            }
        }
    }

    PerfettoWriter::new(trace, event_names.to_vec())?.write_raw_series(&remapped, &thread_meta)?;

    Ok(())
}

pub fn sweep(
    library: Option<PathBuf>,
    config: ResolvedConfig,
    trace: PathBuf,
    matrix: Option<PathBuf>,
    quiet: bool,
    target: Vec<String>,
) -> std::io::Result<()> {
    let lib = load_library(library)?;
    let all_ids: Vec<u32> = (0..lib.events.len() as u32).collect();

    // Reserve one counter slot per batch for the anchor event (retired instructions).
    // Batches become [anchor, user_0, .., user_{MAX_COUNTERS-2}] instead of MAX_COUNTERS
    // user events.
    let anchor_name = "ex_ret_instr";
    let anchor_id: u32 = lib
        .events
        .iter()
        .position(|e| e.name == anchor_name)
        .expect("ex_ret_instr not found in event library") as u32;
    let user_ids: Vec<u32> = all_ids
        .iter()
        .copied()
        .filter(|&id| id != anchor_id)
        .collect();
    let batches: Vec<Vec<u32>> = user_ids
        .chunks(MAX_COUNTERS - 1)
        .map(|c| {
            std::iter::once(anchor_id)
                .chain(c.iter().copied())
                .collect()
        })
        .collect();
    let num_batches = batches.len();
    tracing::info!(
        "Sweep: {} events (+ anchor {}) across {} runs",
        user_ids.len(),
        anchor_name,
        num_batches
    );

    let registry = EventRegistry::new(lib.clone());
    let event_names: Vec<String> = all_ids
        .iter()
        .map(|&id| registry.get_event_name(id).to_string())
        .collect();

    let shared_maps: Rc<RefCell<BatchMaps>> = Rc::new(RefCell::new(Vec::new()));

    let mut sinks: Vec<Box<dyn OutputSink>> = Vec::new();
    sinks.push(Box::new(PerfettoSink::new(
        &trace,
        event_names.clone(),
        0, // emit every quantum
    )?));
    if let Some(matrix_path) = matrix {
        sinks.push(Box::new(MatrixSink::new(
            matrix_path,
            event_names.clone(),
            config.q_sample_ns,
        )));
    }
    sinks.push(Box::new(TidCollectorSink {
        maps: Rc::clone(&shared_maps),
    }));
    let shared_totals: Rc<RefCell<(u64, u64)>> = Rc::new(RefCell::new((0, 0)));
    sinks.push(Box::new(InstructionTrackerSink {
        anchor_id,
        totals: Rc::clone(&shared_totals),
    }));

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta_precise})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb
    };

    for (batch_idx, batch) in batches.iter().enumerate() {
        let registry = EventRegistry::new(lib.clone());
        let counter_names = batch
            .iter()
            .map(|&id| registry.get_event_name(id))
            .collect::<Vec<_>>()
            .join(", ");
        pb.set_message(counter_names);

        for s in &mut sinks {
            s.begin_batch(batch_idx as u32, batch);
        }

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

        let source = HardwareSampleSource::new(pid, registry, None, config.q_sample_ns)
            .expect("Failed to create hardware source");

        let mut profiler = ProfilerBuilder::new()
            .source(source)
            .scheduler(FixedScheduler::new(batch.clone()), batch.clone())
            .map_err(|e| std::io::Error::other(e.to_string()))?
            .estimator(PropagateEstimator::new())
            .sinks(&mut sinks)
            .build();

        syscalls::ptrace_detach(pid)?;

        let quantum_dur = Duration::from_nanos(config.q_schedule_ns);
        while child
            .try_wait()
            .expect("Failed to wait for child")
            .is_none()
        {
            profiler.step();
            thread::sleep(quantum_dur);
        }
        child.wait().unwrap();
        drop(profiler);

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Compute the global instruction rate and configure normalization in all sinks.
    let (total_count, total_dur) = *shared_totals.borrow();
    let global_ref_rate = if total_dur > 0 {
        total_count as f64 / total_dur as f64
    } else {
        1.0
    };
    for sink in &mut sinks {
        sink.set_anchor(anchor_id, global_ref_rate);
    }

    sink::finish_sinks(&mut sinks);
    // Drop sinks before reopening the trace path so PerfettoSink's BufWriter<File> is closed.
    drop(sinks);

    let batch_maps = Rc::try_unwrap(shared_maps).unwrap().into_inner();
    remap_sweep_tids(
        &trace,
        &batch_maps,
        &event_names,
        anchor_name,
        global_ref_rate,
    )?;

    tracing::info!("Sweep complete. Trace written to {:?}", trace);

    Ok(())
}
