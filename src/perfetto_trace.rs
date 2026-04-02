use crate::event_registry::EventId;
use crate::virtual_counter::VirtualCounterState;
use perfetto_protos::counter_descriptor::CounterDescriptor;
use perfetto_protos::trace_packet::TracePacket;
use perfetto_protos::track_descriptor::TrackDescriptor;
use perfetto_protos::track_event::TrackEvent;
use protobuf::Message;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// UUID base offset to avoid UUID 0 (which is the implicit global track).
const UUID_BASE: u64 = 1000;

/// Writes Perfetto trace files containing VCS rate and uncertainty counter tracks.
///
/// The output file is a valid `.perfetto-trace` — a sequence of length-prefixed
/// `TracePacket` messages wrapped in the `Trace` container wire format.
pub struct PerfettoWriter {
    writer: BufWriter<File>,
    event_names: Vec<String>,
}

impl PerfettoWriter {
    pub fn new(path: impl AsRef<Path>, event_names: Vec<String>) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            writer,
            event_names,
        })
    }

    /// Emit TrackDescriptor packets for each event's rate and uncertainty tracks.
    pub fn register_tracks(&mut self) -> std::io::Result<()> {
        let event_names = self.event_names.clone();
        for (i, name) in event_names.iter().enumerate() {
            // Rate track
            {
                let mut counter = CounterDescriptor::new();
                counter.set_unit_name("events/ns".to_string());

                let mut desc = TrackDescriptor::new();
                desc.set_uuid(rate_uuid(i as u32));
                desc.set_name(format!("{}/rate", name));
                desc.counter = protobuf::MessageField::some(counter);

                self.write_track_descriptor_packet(desc)?;
            }

            // Uncertainty track
            {
                let counter = CounterDescriptor::new();

                let mut desc = TrackDescriptor::new();
                desc.set_uuid(uncertainty_uuid(i as u32));
                desc.set_name(format!("{}/uncertainty", name));
                desc.counter = protobuf::MessageField::some(counter);

                self.write_track_descriptor_packet(desc)?;
            }
        }

        Ok(())
    }

    /// Emit counter values for all non-default-state events at the given timestamp.
    pub fn emit_step(
        &mut self,
        timestamp_ns: u64,
        vcs: &VirtualCounterState,
        _active_set: &[EventId],
    ) -> std::io::Result<()> {
        for (i, est) in vcs.all_estimates().iter().enumerate() {
            // Skip never-observed counters still at defaults
            if est.rate == 0.0 && est.uncertainty == 1.0 && est.sample_count == 0 {
                continue;
            }

            self.write_counter_packet(timestamp_ns, rate_uuid(i as u32), est.rate)?;
            self.write_counter_packet(timestamp_ns, uncertainty_uuid(i as u32), est.uncertainty)?;
        }

        Ok(())
    }

    fn write_track_descriptor_packet(&mut self, desc: TrackDescriptor) -> std::io::Result<()> {
        let mut packet = TracePacket::new();
        packet.set_track_descriptor(desc);
        packet.set_trusted_packet_sequence_id(1);
        self.write_trace_packet(&packet)
    }

    fn write_counter_packet(
        &mut self,
        timestamp_ns: u64,
        track_uuid: u64,
        value: f64,
    ) -> std::io::Result<()> {
        use perfetto_protos::track_event::track_event::Type;

        let mut event = TrackEvent::new();
        event.set_type(Type::TYPE_COUNTER);
        event.set_track_uuid(track_uuid);
        event.set_double_counter_value(value);

        let mut packet = TracePacket::new();
        packet.set_timestamp(timestamp_ns);
        packet.set_trusted_packet_sequence_id(1);
        packet.set_track_event(event);
        self.write_trace_packet(&packet)
    }

    /// Write a single TracePacket in the Trace container wire format.
    ///
    /// A `.perfetto-trace` file is a serialized `Trace` protobuf, which is just
    /// `repeated TracePacket packet = 1`. Each packet is written as:
    ///   field tag (0x0A = field 1, wire type LEN) + varint length + packet bytes
    fn write_trace_packet(&mut self, packet: &TracePacket) -> std::io::Result<()> {
        let bytes = packet.write_to_bytes().map_err(std::io::Error::other)?;

        // Trace.packet = field 1, wire type 2 (LEN) -> tag byte 0x0A
        self.writer.write_all(&[0x0A])?;
        // Varint-encode the length
        write_varint(&mut self.writer, bytes.len() as u64)?;
        self.writer.write_all(&bytes)?;

        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl Drop for PerfettoWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

fn rate_uuid(event_id: u32) -> u64 {
    UUID_BASE + (event_id as u64) * 2
}

fn uncertainty_uuid(event_id: u32) -> u64 {
    UUID_BASE + (event_id as u64) * 2 + 1
}

/// Write a u64 as a protobuf varint.
fn write_varint(w: &mut impl Write, mut value: u64) -> std::io::Result<()> {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            w.write_all(&[byte])?;
            return Ok(());
        }
        w.write_all(&[byte | 0x80])?;
    }
}
