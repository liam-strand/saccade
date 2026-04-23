use crate::event::EventId;
use crate::perfetto::PerfettoWriter;
use crate::quantum::Quantum;
use crate::sample::TASK_COMM_LEN;
use crate::sink::OutputSink;
use crate::state::StateEstimator;
use std::collections::HashMap;
use std::path::Path;

/// Perfetto trace sink. Emits per-(thread, event) counter tracks from the
/// state estimator's current snapshot.
///
/// Output cadence is decoupled from the scheduling quantum: the sink only
/// writes a snapshot when at least `output_period_ns` has elapsed since the
/// previous write. `output_period_ns = 0` means "emit every quantum".
pub struct PerfettoSink {
    writer: PerfettoWriter,
    /// tid → (tgid, task_name) — accumulated across quantums for track registration.
    thread_meta: HashMap<u32, (u32, String)>,
    output_period_ns: u64,
    last_emit_ns: Option<u64>,
}

impl PerfettoSink {
    pub fn new(
        path: impl AsRef<Path>,
        event_names: Vec<String>,
        output_period_ns: u64,
    ) -> std::io::Result<Self> {
        let writer = PerfettoWriter::new(path, event_names)?;
        Ok(Self {
            writer,
            thread_meta: HashMap::new(),
            output_period_ns,
            last_emit_ns: None,
        })
    }
}

impl OutputSink for PerfettoSink {
    fn emit(
        &mut self,
        quantum: &Quantum,
        estimator: &dyn StateEstimator,
        _active_set: &[EventId],
    ) -> std::io::Result<()> {
        // Update thread metadata from this quantum's raw samples.
        // tid=0 is the kernel idle task; it is never a meaningful profiling target.
        for s in quantum.samples() {
            if s.tid == 0 {
                continue;
            }
            self.thread_meta.entry(s.tid).or_insert_with(|| {
                let task_len = s.task.iter().position(|&c| c == 0).unwrap_or(TASK_COMM_LEN);
                let task_name = String::from_utf8_lossy(&s.task[..task_len]).into_owned();
                (s.pid, task_name)
            });
        }

        let ts = quantum.timestamp_ns();
        let should_emit = match self.last_emit_ns {
            None => true,
            Some(last) => ts.saturating_sub(last) >= self.output_period_ns,
        };
        if !should_emit {
            return Ok(());
        }
        self.last_emit_ns = Some(ts);

        self.writer
            .emit_estimator_snapshot(ts, estimator, &self.thread_meta)
    }

    fn finish(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}
