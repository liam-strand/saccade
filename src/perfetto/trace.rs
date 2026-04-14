use crate::event::EventId;
use crate::quantum::EventAggregate;
use crate::virtual_counter::VirtualCounterState;
use perfetto_protos::counter_descriptor::CounterDescriptor;
use perfetto_protos::process_descriptor::ProcessDescriptor;
use perfetto_protos::thread_descriptor::ThreadDescriptor;
use perfetto_protos::trace_packet::TracePacket;
use perfetto_protos::track_descriptor::TrackDescriptor;
use perfetto_protos::track_event::TrackEvent;
use protobuf::Message;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// UUID base offset to avoid UUID 0 (which is the implicit global track).
const UUID_BASE: u64 = 1000;
/// UUID base for dynamically-allocated per-thread tracks.
const THREAD_UUID_BASE: u64 = 1_000_000;

/// Writes Perfetto trace files containing VCS rate and uncertainty counter tracks.
///
/// The output file is a valid `.perfetto-trace` — a sequence of length-prefixed
/// `TracePacket` messages wrapped in the `Trace` container wire format.
pub struct PerfettoWriter {
    writer: BufWriter<File>,
    event_names: Vec<String>,
    /// Next UUID to allocate for per-thread tracks.
    next_thread_uuid: u64,
    /// tgid → process track uuid
    process_uuids: HashMap<u32, u64>,
    /// tid → thread track uuid
    thread_uuids: HashMap<u32, u64>,
    /// (tid, event_id) → (rate_uuid, uncertainty_uuid)
    thread_counter_uuids: HashMap<(u32, u32), (u64, u64)>,
}

impl PerfettoWriter {
    pub fn new(path: impl AsRef<Path>, event_names: Vec<String>) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            writer,
            event_names,
            next_thread_uuid: THREAD_UUID_BASE,
            process_uuids: HashMap::new(),
            thread_uuids: HashMap::new(),
            thread_counter_uuids: HashMap::new(),
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
                let mut counter = CounterDescriptor::new();
                counter.set_unit_name("uncertainty".to_string());

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

    /// Lazily allocate a UUID for a process track; emit TrackDescriptor on first call.
    fn ensure_process_track(&mut self, tgid: u32, name: &str) -> std::io::Result<u64> {
        if let Some(&uuid) = self.process_uuids.get(&tgid) {
            return Ok(uuid);
        }
        let uuid = self.next_thread_uuid;
        self.next_thread_uuid += 1;
        self.process_uuids.insert(tgid, uuid);

        let mut proc_desc = ProcessDescriptor::new();
        proc_desc.set_pid(tgid as i32);
        proc_desc.set_process_name(name.to_string());

        let mut desc = TrackDescriptor::new();
        desc.set_uuid(uuid);
        desc.set_name(name.to_string());
        desc.process = protobuf::MessageField::some(proc_desc);

        self.write_track_descriptor_packet(desc)?;
        Ok(uuid)
    }

    /// Lazily allocate a UUID for a thread track; emit TrackDescriptor on first call.
    fn ensure_thread_track(&mut self, tgid: u32, tid: u32, task: &str) -> std::io::Result<u64> {
        if let Some(&uuid) = self.thread_uuids.get(&tid) {
            return Ok(uuid);
        }
        let process_uuid = self.ensure_process_track(tgid, task)?;
        let uuid = self.next_thread_uuid;
        self.next_thread_uuid += 1;
        self.thread_uuids.insert(tid, uuid);

        let mut thread_desc = ThreadDescriptor::new();
        thread_desc.set_pid(tgid as i32);
        thread_desc.set_tid(tid as i32);
        thread_desc.set_thread_name(task.to_string());

        let mut desc = TrackDescriptor::new();
        desc.set_uuid(uuid);
        desc.set_name(task.to_string());
        desc.set_parent_uuid(process_uuid);
        desc.thread = protobuf::MessageField::some(thread_desc);

        self.write_track_descriptor_packet(desc)?;
        Ok(uuid)
    }

    /// Lazily allocate UUIDs for per-thread rate/uncertainty counter tracks.
    fn ensure_thread_counter_tracks(
        &mut self,
        tid: u32,
        event_id: u32,
        thread_uuid: u64,
    ) -> std::io::Result<(u64, u64)> {
        if let Some(&uuids) = self.thread_counter_uuids.get(&(tid, event_id)) {
            return Ok(uuids);
        }
        let rate_uuid = self.next_thread_uuid;
        self.next_thread_uuid += 1;
        let uncertainty_uuid = self.next_thread_uuid;
        self.next_thread_uuid += 1;
        self.thread_counter_uuids
            .insert((tid, event_id), (rate_uuid, uncertainty_uuid));

        let event_name = self
            .event_names
            .get(event_id as usize)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        for (uuid, suffix, unit) in [
            (rate_uuid, "rate", "events/ns"),
            (uncertainty_uuid, "uncertainty", "uncertainty"),
        ] {
            let mut counter = CounterDescriptor::new();
            counter.set_unit_name(unit.to_string());

            let mut desc = TrackDescriptor::new();
            desc.set_uuid(uuid);
            desc.set_name(format!("{}/{}", event_name, suffix));
            desc.set_parent_uuid(thread_uuid);
            desc.counter = protobuf::MessageField::some(counter);

            self.write_track_descriptor_packet(desc)?;
        }
        Ok((rate_uuid, uncertainty_uuid))
    }

    /// Emit per-thread counter values for a single quantum step.
    ///
    /// `thread_meta`: tid → (tgid, task_name) — needed to register process/thread tracks.
    pub fn emit_thread_steps(
        &mut self,
        timestamp_ns: u64,
        per_thread: &HashMap<(u32, EventId), EventAggregate>,
        thread_meta: &HashMap<u32, (u32, String)>,
    ) -> std::io::Result<()> {
        for (&(tid, event_id), agg) in per_thread {
            let (tgid, task) = match thread_meta.get(&tid) {
                Some(m) => (m.0, m.1.as_str()),
                None => continue,
            };
            let thread_uuid = self.ensure_thread_track(tgid, tid, task)?;
            let (rate_uuid, uncert_uuid) =
                self.ensure_thread_counter_tracks(tid, event_id, thread_uuid)?;
            self.write_counter_packet(timestamp_ns, rate_uuid, agg.mean_rate)?;
            self.write_counter_packet(timestamp_ns, uncert_uuid, agg.stddev_rate)?;
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

    /// Write pre-collected per-thread time-series data as counter packets.
    ///
    /// `series`: (event_id, tid) → sorted Vec of (timestamp_ns, rate).
    /// `thread_meta`: tid → (tgid, task_name) — used to register process/thread tracks.
    /// Emits per-thread rate and uncertainty (0.0 = observed) packets sorted by timestamp.
    pub fn write_raw_series(
        &mut self,
        series: &HashMap<(u32, u32), Vec<(u64, f64)>>,
        thread_meta: &HashMap<u32, (u32, String)>,
    ) -> std::io::Result<()> {
        // Collect all points with their (event_id, tid) key, sorted by timestamp.
        let mut points: Vec<(u64, u32, u32, f64)> = series
            .iter()
            .flat_map(|(&(event_id, tid), pts)| {
                pts.iter().map(move |&(ts, rate)| (ts, event_id, tid, rate))
            })
            .collect();
        points.sort_unstable_by_key(|&(ts, _, _, _)| ts);

        for (ts, event_id, tid, rate) in points {
            let (tgid, task) = match thread_meta.get(&tid) {
                Some(m) => (m.0, m.1.clone()),
                None => continue,
            };
            let thread_uuid = self.ensure_thread_track(tgid, tid, &task)?;
            let (r_uuid, u_uuid) = self.ensure_thread_counter_tracks(tid, event_id, thread_uuid)?;
            self.write_counter_packet(ts, r_uuid, rate)?;
            self.write_counter_packet(ts, u_uuid, 0.0)?;
        }
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

/// Decode a protobuf varint from a byte slice.
/// Returns `(value, bytes_consumed)` or `None` if the slice is truncated.
pub(crate) fn read_varint(data: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

/// Parse a `.perfetto-trace` byte slice into `TracePacket`s.
///
/// The wire format is the `Trace` protobuf container:
///   repeated TracePacket packet = 1;
/// Each packet is encoded as: tag 0x0A + varint length + packet bytes.
pub(crate) fn read_trace_packets(data: &[u8]) -> std::io::Result<Vec<TracePacket>> {
    let mut packets = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        if data[pos] != 0x0A {
            return Err(std::io::Error::other(format!(
                "Unexpected tag byte 0x{:02X} at offset {}",
                data[pos], pos
            )));
        }
        pos += 1;
        let (len, consumed) = read_varint(&data[pos..])
            .ok_or_else(|| std::io::Error::other("Truncated varint in trace file"))?;
        pos += consumed;
        let end = pos + len as usize;
        if end > data.len() {
            return Err(std::io::Error::other("Packet extends beyond end of file"));
        }
        let packet =
            TracePacket::parse_from_bytes(&data[pos..end]).map_err(std::io::Error::other)?;
        packets.push(packet);
        pos = end;
    }
    Ok(packets)
}
