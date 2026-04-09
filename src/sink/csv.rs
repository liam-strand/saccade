use crate::event::EventId;
use crate::quantum::Quantum;
use crate::sink::OutputSink;
use crate::virtual_counter::VirtualCounterState;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// CSV sink. Writes one row per `RawSample` with raw counts and durations.
/// Rate computation is left to the consumer; this preserves the full raw data.
pub struct CsvSink {
    writer: BufWriter<File>,
}

impl CsvSink {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        writeln!(
            writer,
            "timestamp_ns,duration_ns,cpu_id,pid,event_id,count,task"
        )?;
        Ok(Self { writer })
    }
}

impl OutputSink for CsvSink {
    fn emit(
        &mut self,
        quantum: &Quantum,
        _vcs: &VirtualCounterState,
        _active_set: &[EventId],
    ) -> std::io::Result<()> {
        for s in quantum.samples() {
            let task_len = s.task.iter().position(|&c| c == 0).unwrap_or(s.task.len());
            let task_name = String::from_utf8_lossy(&s.task[..task_len]);
            writeln!(
                self.writer,
                "{},{},{},{},{},{},{}",
                s.timestamp_ns, s.duration_ns, s.cpu_id, s.pid, s.event_id, s.count, task_name,
            )?;
        }
        Ok(())
    }

    fn finish(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}
