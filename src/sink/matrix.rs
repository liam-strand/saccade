//! HDF5 matrix output sink that bins per-thread event rates over time.

use crate::event::EventId;
use crate::quantum::Quantum;
use crate::sample::TASK_COMM_LEN;
use crate::sink::OutputSink;
use crate::state::StateEstimator;
use hdf5::types::VarLenUnicode;
use ndarray::Array2;
use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::str::FromStr;

/// Accumulated time-series data for one synthetic thread.
struct ThreadData {
    /// Process name from the kernel task comm string.
    task_name: String,
    /// Thread group ID (process ID) of the real thread.
    tgid: u32,
    /// Per-event list of `(rel_ts_ns, rate)` observations collected across quantums.
    /// In sweep mode the anchor event's rates are events/ns; all other events are stored
    /// as events/instruction and scaled to events/ns in `finish()`.
    series: HashMap<EventId, Vec<(u64, f64)>>,
}

/// Writes an HDF5 file with one `thread_<N>/rates` dataset per observed thread,
/// where `rates` is an `[n_events × n_timesteps]` f32 matrix of mean event rates (events/ns).
///
/// In sweep mode, rates are instruction-normalized: non-anchor events are stored as
/// `count / anchor_count` (events/instruction) during collection, then scaled by the global
/// instruction rate in `finish()` so the final unit is events/ns across all batches.
/// The anchor event (`ex_ret_instr`) is written as plain events/ns throughout.
pub struct MatrixSink {
    /// Output HDF5 file path.
    path: PathBuf,
    /// Width of each time bin in nanoseconds.
    dt_ns: u64,
    /// Total number of hardware events tracked.
    n_events: u32,
    /// Human-readable names for each event, indexed by `EventId`.
    event_names: Vec<String>,
    /// Maps each event index to the batch that produced its samples; -1 if not yet sampled.
    event_to_batch: Vec<i32>,

    /// Accumulated per-synthetic-TID data, keyed by synthetic TID.
    threads: HashMap<u32, ThreadData>,

    // Cross-batch synthetic-TID assignment
    /// Maps `(task_name, within-name ordinal)` to a stable synthetic TID across batches.
    name_idx_to_synthetic: HashMap<(String, u32), u32>,
    /// Monotonically increasing counter used to allocate the next synthetic TID.
    next_synthetic_tid: u32,

    // Per-batch transient state (reset in begin_batch)
    /// Batch ID currently being accumulated.
    current_batch_id: u32,
    /// Events active in the current batch.
    current_batch_events: Vec<EventId>,
    /// Maps real TIDs to synthetic TIDs for the duration of the current batch.
    batch_real_to_synthetic: HashMap<u32, u32>,
    /// Counts how many distinct real TIDs share each task name within the current batch.
    batch_name_counters: HashMap<String, u32>,
    /// Timestamp of the first sample seen in the current batch, used to compute relative times.
    batch_t0: Option<u64>,

    /// Anchor event ID for instruction-count normalization (None in non-sweep runs).
    anchor_id: Option<EventId>,
    /// Global instruction rate (instructions/ns) applied in finish() to convert
    /// events/instruction back to events/ns (None in non-sweep runs).
    global_ref_rate: Option<f64>,
}

impl MatrixSink {
    /// Create a new `MatrixSink` that will write to `path` with `dt_ns`-wide time bins.
    pub fn new(path: PathBuf, event_names: Vec<String>, dt_ns: u64) -> Self {
        let n_events = event_names.len() as u32;
        Self {
            path,
            dt_ns,
            n_events,
            event_names,
            event_to_batch: vec![-1_i32; n_events as usize],
            threads: HashMap::new(),
            name_idx_to_synthetic: HashMap::new(),
            next_synthetic_tid: 1,
            current_batch_id: 0,
            current_batch_events: Vec::new(),
            batch_real_to_synthetic: HashMap::new(),
            batch_name_counters: HashMap::new(),
            batch_t0: None,
            anchor_id: None,
            global_ref_rate: None,
        }
    }
}

impl OutputSink for MatrixSink {
    fn set_anchor(&mut self, anchor_id: EventId, global_ref_rate: f64) {
        self.anchor_id = Some(anchor_id);
        self.global_ref_rate = Some(global_ref_rate);
    }

    fn begin_batch(&mut self, batch_id: u32, events: &[EventId]) {
        self.current_batch_id = batch_id;
        self.current_batch_events = events.to_vec();
        for &eid in events {
            if (eid as usize) < self.event_to_batch.len() {
                self.event_to_batch[eid as usize] = batch_id as i32;
            }
        }
        self.batch_real_to_synthetic.clear();
        self.batch_name_counters.clear();
        self.batch_t0 = None;
    }

    fn emit(
        &mut self,
        quantum: &Quantum,
        _estimator: &dyn StateEstimator,
        _active_set: &[EventId],
    ) -> io::Result<()> {
        // In sweep mode, build a (cpu_id, timestamp_ns, tid) → anchor_count lookup so
        // each sample can be normalized by the co-measured instruction count.
        // Samples from the same BPF wire event share identical (cpu_id, timestamp_ns, tid).
        let anchor_counts: HashMap<(u32, u64, u32), u64> =
            if let Some(anchor_id) = self.anchor_id {
                quantum
                    .samples()
                    .iter()
                    .filter(|s| s.event_id == anchor_id && s.duration_ns > 0)
                    .map(|s| ((s.cpu_id, s.timestamp_ns, s.tid), s.count))
                    .collect()
            } else {
                HashMap::new()
            };

        for s in quantum.samples() {
            if s.duration_ns == 0 {
                continue;
            }

            let t0 = *self.batch_t0.get_or_insert(s.timestamp_ns);
            let rel_ts = s.timestamp_ns.saturating_sub(t0);

            let task_len = s.task.iter().position(|&c| c == 0).unwrap_or(TASK_COMM_LEN);
            let task_name = String::from_utf8_lossy(&s.task[..task_len]).into_owned();

            let synthetic_tid = *self
                .batch_real_to_synthetic
                .entry(s.tid)
                .or_insert_with(|| {
                    let counter = self
                        .batch_name_counters
                        .entry(task_name.clone())
                        .or_insert(0);
                    let within_name_idx = *counter;
                    *counter += 1;
                    *self
                        .name_idx_to_synthetic
                        .entry((task_name.clone(), within_name_idx))
                        .or_insert_with(|| {
                            let id = self.next_synthetic_tid;
                            self.next_synthetic_tid += 1;
                            id
                        })
                });

            let td = self
                .threads
                .entry(synthetic_tid)
                .or_insert_with(|| ThreadData {
                    task_name: task_name.clone(),
                    tgid: s.pid,
                    series: HashMap::new(),
                });

            let rate = match self.anchor_id {
                Some(anchor_id) if s.event_id != anchor_id => {
                    // Non-anchor in sweep mode: events/instruction.
                    // global_ref_rate is applied in finish() to convert to events/ns.
                    let ac = anchor_counts
                        .get(&(s.cpu_id, s.timestamp_ns, s.tid))
                        .copied()
                        .unwrap_or(0);
                    if ac > 0 {
                        s.count as f64 / ac as f64
                    } else {
                        s.count as f64 / s.duration_ns as f64
                    }
                }
                _ => s.count as f64 / s.duration_ns as f64,
            };
            td.series
                .entry(s.event_id)
                .or_default()
                .push((rel_ts, rate));
        }
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        let dt = self.dt_ns.max(1);

        // Convert events/instruction → events/ns for non-anchor events by applying
        // the global instruction rate computed across all sweep batches.
        if let (Some(anchor_id), Some(ref_rate)) = (self.anchor_id, self.global_ref_rate) {
            for td in self.threads.values_mut() {
                for (&event_id, pts) in td.series.iter_mut() {
                    if event_id != anchor_id {
                        for (_, rate) in pts.iter_mut() {
                            *rate *= ref_rate;
                        }
                    }
                }
            }
        }

        // Compute per-thread max rel_ts to determine n_timesteps
        let mut per_thread_max_ts: HashMap<u32, u64> = HashMap::new();
        for (&syn_tid, td) in &self.threads {
            let max_ts = td
                .series
                .values()
                .flat_map(|pts| pts.iter().map(|&(ts, _)| ts))
                .max()
                .unwrap_or(0);
            per_thread_max_ts.insert(syn_tid, max_ts);
        }

        let file = hdf5::File::create(&self.path).map_err(|e| io::Error::other(e.to_string()))?;

        // Root attributes
        file.new_attr::<u64>()
            .shape(())
            .create("dt_ns")
            .and_then(|a| a.write_scalar(&dt))
            .map_err(|e| io::Error::other(e.to_string()))?;
        file.new_attr::<u32>()
            .shape(())
            .create("n_events")
            .and_then(|a| a.write_scalar(&self.n_events))
            .map_err(|e| io::Error::other(e.to_string()))?;

        // /event_names — variable-length UTF-8 strings
        {
            let vlu: Vec<VarLenUnicode> = self
                .event_names
                .iter()
                .map(|s| VarLenUnicode::from_str(s).unwrap_or_default())
                .collect();
            let n = vlu.len();
            file.new_dataset::<VarLenUnicode>()
                .shape((n,))
                .create("event_names")
                .and_then(|ds| ds.write(&vlu))
                .map_err(|e| io::Error::other(e.to_string()))?;
        }

        // /batch_id — which batch produced each event row
        {
            let n = self.event_to_batch.len();
            file.new_dataset::<i32>()
                .shape((n,))
                .create("batch_id")
                .and_then(|ds| ds.write(&self.event_to_batch))
                .map_err(|e| io::Error::other(e.to_string()))?;
        }

        // /thread_<synthetic_tid>/rates [n_events × n_timesteps]
        let mut syn_tids: Vec<u32> = self.threads.keys().copied().collect();
        syn_tids.sort_unstable();

        for syn_tid in syn_tids {
            let td = &self.threads[&syn_tid];
            let max_ts = per_thread_max_ts[&syn_tid];
            let n_timesteps = (max_ts / dt + 1) as usize;
            let n_events = self.n_events as usize;

            // Build n_events × n_timesteps matrix, NaN-initialised.
            let mut rates = Array2::<f32>::from_elem((n_events, n_timesteps), f32::NAN);

            for (&event_id, pts) in &td.series {
                if event_id as usize >= n_events {
                    continue;
                }
                // Bin: for each timestep collect sample rates, then mean them.
                let mut bins: Vec<(f64, u32)> = vec![(0.0, 0); n_timesteps];
                for &(ts, rate) in pts {
                    let bin = (ts / dt) as usize;
                    if bin < n_timesteps {
                        bins[bin].0 += rate;
                        bins[bin].1 += 1;
                    }
                }
                for (t, (sum, count)) in bins.into_iter().enumerate() {
                    if count > 0 {
                        rates[[event_id as usize, t]] = (sum / count as f64) as f32;
                    }
                }
            }

            let group_name = format!("thread_{syn_tid}");
            let group = file
                .create_group(&group_name)
                .map_err(|e| io::Error::other(e.to_string()))?;

            // Group attributes
            let task_vlu = VarLenUnicode::from_str(&td.task_name).unwrap_or_default();
            group
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create("task_name")
                .and_then(|a| a.write_scalar(&task_vlu))
                .map_err(|e| io::Error::other(e.to_string()))?;
            group
                .new_attr::<u32>()
                .shape(())
                .create("tgid")
                .and_then(|a| a.write_scalar(&td.tgid))
                .map_err(|e| io::Error::other(e.to_string()))?;
            group
                .new_attr::<u64>()
                .shape(())
                .create("n_timesteps")
                .and_then(|a| a.write_scalar(&(n_timesteps as u64)))
                .map_err(|e| io::Error::other(e.to_string()))?;

            // Write the rates matrix
            group
                .new_dataset_builder()
                .with_data(&rates)
                .create("rates")
                .map_err(|e| io::Error::other(e.to_string()))?;
        }

        Ok(())
    }
}
