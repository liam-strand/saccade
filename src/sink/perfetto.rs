use crate::event::EventId;
use crate::perfetto::PerfettoWriter;
use crate::quantum::Quantum;
use crate::sample::TASK_COMM_LEN;
use crate::sink::OutputSink;
use crate::state::StateEstimator;
use std::collections::HashMap;
use std::path::Path;

/// Perfetto trace sink. Emits VCS rate/uncertainty aggregate tracks plus per-thread counter tracks.
pub struct PerfettoSink {
    writer: PerfettoWriter,
    /// tid → (tgid, task_name) — accumulated across quantums for track registration.
    thread_meta: HashMap<u32, (u32, String)>,
}

impl PerfettoSink {
    pub fn new(path: impl AsRef<Path>, event_names: Vec<String>) -> std::io::Result<Self> {
        let mut writer = PerfettoWriter::new(path, event_names)?;
        writer.register_tracks()?;
        Ok(Self {
            writer,
            thread_meta: HashMap::new(),
        })
    }
}

impl OutputSink for PerfettoSink {
    fn emit(
        &mut self,
        quantum: &Quantum,
        estimator: &dyn StateEstimator,
        active_set: &[EventId],
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

        // Emit aggregate VCS tracks (unchanged).
        self.writer
            .emit_step(quantum.timestamp_ns(), estimator, active_set)?;

        // Emit per-thread counter tracks.
        self.writer.emit_thread_steps(
            quantum.timestamp_ns(),
            quantum.per_thread_aggregates(),
            &self.thread_meta,
        )
    }

    fn finish(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}
