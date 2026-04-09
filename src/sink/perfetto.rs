use crate::event::EventId;
use crate::perfetto::PerfettoWriter;
use crate::quantum::Quantum;
use crate::sink::OutputSink;
use crate::virtual_counter::VirtualCounterState;
use std::path::Path;

/// Perfetto trace sink. Emits VCS rate and uncertainty counter tracks.
pub struct PerfettoSink {
    writer: PerfettoWriter,
}

impl PerfettoSink {
    pub fn new(path: impl AsRef<Path>, event_names: Vec<String>) -> std::io::Result<Self> {
        let mut writer = PerfettoWriter::new(path, event_names)?;
        writer.register_tracks()?;
        Ok(Self { writer })
    }
}

impl OutputSink for PerfettoSink {
    fn emit(
        &mut self,
        quantum: &Quantum,
        vcs: &VirtualCounterState,
        active_set: &[EventId],
    ) -> std::io::Result<()> {
        self.writer
            .emit_step(quantum.timestamp_ns(), vcs, active_set)
    }

    fn finish(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}
