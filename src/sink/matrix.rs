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

struct ThreadData {
    task_name: String,
    tgid: u32,
    /// Raw (rel_ts_ns, rate) per event. Outer index = event_id (same as global).
    series: HashMap<EventId, Vec<(u64, f64)>>,
}

pub struct MatrixSink {
    path: PathBuf,
    dt_ns: u64,
    n_events: u32,
    event_names: Vec<String>,
    /// Which batch_id produced each event row (-1 = not yet sampled).
    event_to_batch: Vec<i32>,

    threads: HashMap<u32, ThreadData>,

    // Cross-batch synthetic-TID assignment
    name_idx_to_synthetic: HashMap<(String, u32), u32>,
    next_synthetic_tid: u32,

    // Per-batch transient state (reset in begin_batch)
    current_batch_id: u32,
    current_batch_events: Vec<EventId>,
    batch_real_to_synthetic: HashMap<u32, u32>,
    batch_name_counters: HashMap<String, u32>,
    batch_t0: Option<u64>,
}

impl MatrixSink {
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
        }
    }
}

impl OutputSink for MatrixSink {
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

            let rate = s.count as f64 / s.duration_ns as f64;
            td.series
                .entry(s.event_id)
                .or_default()
                .push((rel_ts, rate));
        }
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        let dt = self.dt_ns.max(1);

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
