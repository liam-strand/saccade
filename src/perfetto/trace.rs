//! Writing Perfetto proto binary trace files from Saccade counter samples.

use crate::event::EventId;
use crate::state::StateEstimator;
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

/// Starting value for sequentially-allocated UUIDs for all dynamically-created tracks (process, thread, and counter).
const THREAD_UUID_BASE: u64 = 1_000_000;

/// Writes Perfetto trace files containing per-(thread, event) rate counter tracks.
///
/// The output file is a valid `.perfetto-trace` — a sequence of length-prefixed
/// `TracePacket` messages wrapped in the `Trace` container wire format.
pub struct PerfettoWriter {
    /// Buffered writer for the output `.perfetto-trace` file.
    writer: BufWriter<File>,
    /// Ordered list of hardware event names, indexed by `EventId`.
    event_names: Vec<String>,
    /// Monotonically increasing UUID counter for all dynamically-allocated tracks.
    next_thread_uuid: u64,
    /// Maps tgid → UUID of the corresponding process track.
    process_uuids: HashMap<u32, u64>,
    /// Maps tid → UUID of the corresponding thread track.
    thread_uuids: HashMap<u32, u64>,
    /// Maps (tid, event_id) → UUID of the rate counter track for that pair.
    thread_counter_uuids: HashMap<(u32, u32), u64>,
    /// Maps (tid, event_id) → UUID of the uncertainty counter track for that pair.
    thread_uncertainty_uuids: HashMap<(u32, u32), u64>,
}

impl PerfettoWriter {
    /// Create a new writer that will serialize traces to `path`, using `event_names` to label counter tracks.
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
            thread_uncertainty_uuids: HashMap::new(),
        })
    }

    /// Emit one rate counter packet and one uncertainty counter packet per tracked (tid, event_id)
    /// pair using the estimator's current snapshot. `thread_meta` maps tid → (tgid, task_name)
    /// and is required to register process/thread tracks; (tid, event_id) pairs whose tid is not
    /// yet in `thread_meta` are skipped.
    ///
    /// The rate track is named `{event}/rate` and carries `est.rate` (events/ns).
    /// The uncertainty track is named `{event}/uncertainty` and carries `est.uncertainty` ∈ [0, 1],
    /// where 0 = fully confident and 1 = completely unknown.
    pub fn emit_estimator_snapshot(
        &mut self,
        timestamp_ns: u64,
        estimator: &dyn StateEstimator,
        thread_meta: &HashMap<u32, (u32, String)>,
    ) -> std::io::Result<()> {
        for (&(tid, event_id), est) in estimator.all_estimates() {
            let (tgid, task) = match thread_meta.get(&tid) {
                Some(m) => (m.0, m.1.as_str()),
                None => continue,
            };
            let thread_uuid = self.ensure_thread_track(tgid, tid, task)?;
            let rate_uuid = self.ensure_thread_counter_tracks(tid, event_id, thread_uuid)?;
            self.write_counter_packet(timestamp_ns, rate_uuid, est.rate)?;
            let unc_uuid = self.ensure_thread_uncertainty_track(tid, event_id, thread_uuid)?;
            self.write_counter_packet(timestamp_ns, unc_uuid, est.uncertainty)?;
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
        // Use the thread's task name as the process name only when this is the main thread
        // (tid == tgid). Otherwise fall back to a numeric name so that the process track is
        // not accidentally labeled with an arbitrary worker thread's comm string.
        let process_name;
        let process_name_str: &str = if tid == tgid {
            task
        } else {
            process_name = format!("{}", tgid);
            &process_name
        };
        let process_uuid = self.ensure_process_track(tgid, process_name_str)?;
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

    /// Lazily allocate a UUID for a per-thread rate counter track.
    fn ensure_thread_counter_tracks(
        &mut self,
        tid: u32,
        event_id: u32,
        thread_uuid: u64,
    ) -> std::io::Result<u64> {
        if let Some(&uuid) = self.thread_counter_uuids.get(&(tid, event_id)) {
            return Ok(uuid);
        }
        let rate_uuid = self.next_thread_uuid;
        self.next_thread_uuid += 1;
        self.thread_counter_uuids.insert((tid, event_id), rate_uuid);

        let event_name = self
            .event_names
            .get(event_id as usize)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let mut counter = CounterDescriptor::new();
        counter.set_unit_name("events/ns".to_string());

        let mut desc = TrackDescriptor::new();
        desc.set_uuid(rate_uuid);
        desc.set_name(format!("{}/rate", event_name));
        desc.set_parent_uuid(thread_uuid);
        desc.counter = protobuf::MessageField::some(counter);

        self.write_track_descriptor_packet(desc)?;
        Ok(rate_uuid)
    }

    /// Lazily allocate a UUID for a per-thread uncertainty counter track (dimensionless, [0, 1]).
    fn ensure_thread_uncertainty_track(
        &mut self,
        tid: u32,
        event_id: u32,
        thread_uuid: u64,
    ) -> std::io::Result<u64> {
        if let Some(&uuid) = self.thread_uncertainty_uuids.get(&(tid, event_id)) {
            return Ok(uuid);
        }
        let unc_uuid = self.next_thread_uuid;
        self.next_thread_uuid += 1;
        self.thread_uncertainty_uuids
            .insert((tid, event_id), unc_uuid);

        let event_name = self
            .event_names
            .get(event_id as usize)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let mut counter = CounterDescriptor::new();
        counter.set_unit_name("".to_string()); // dimensionless

        let mut desc = TrackDescriptor::new();
        desc.set_uuid(unc_uuid);
        desc.set_name(format!("{}/uncertainty", event_name));
        desc.set_parent_uuid(thread_uuid);
        desc.counter = protobuf::MessageField::some(counter);

        self.write_track_descriptor_packet(desc)?;
        Ok(unc_uuid)
    }

    /// Wrap a `TrackDescriptor` in a `TracePacket` and write it to the output file.
    fn write_track_descriptor_packet(&mut self, desc: TrackDescriptor) -> std::io::Result<()> {
        let mut packet = TracePacket::new();
        packet.set_track_descriptor(desc);
        packet.set_trusted_packet_sequence_id(1);
        self.write_trace_packet(&packet)
    }

    /// Write a `TYPE_COUNTER` track event packet carrying a double counter value at the given timestamp.
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

    /// Serialize `packet` into the `Trace` container wire format: tag `0x0A` + varint length + packet bytes.
    fn write_trace_packet(&mut self, packet: &TracePacket) -> std::io::Result<()> {
        let bytes = packet.write_to_bytes().map_err(std::io::Error::other)?;

        // Trace.packet = field 1, wire type 2 (LEN) -> tag byte 0x0A
        self.writer.write_all(&[0x0A])?;
        // Varint-encode the length
        write_varint(&mut self.writer, bytes.len() as u64)?;
        self.writer.write_all(&bytes)?;

        Ok(())
    }

    /// Write pre-collected `(event_id, tid)` time-series as counter packets, registering tracks via `thread_meta`.
    pub fn write_raw_series(
        &mut self,
        series: &HashMap<(EventId, u32), Vec<(u64, f64)>>,
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
                Some(m) => (m.0, m.1.as_str()),
                None => continue,
            };
            let thread_uuid = self.ensure_thread_track(tgid, tid, task)?;
            let r_uuid = self.ensure_thread_counter_tracks(tid, event_id, thread_uuid)?;
            self.write_counter_packet(ts, r_uuid, rate)?;
        }
        Ok(())
    }

    /// Flush the internal buffer to disk; also called automatically on drop.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl Drop for PerfettoWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
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

/// Walk a `.perfetto-trace` byte slice and call `f` on each decoded `TracePacket`, dropping it before moving to the next.
///
/// Prefer this over `read_trace_packets` when you only need to extract a subset of fields —
/// only one packet's heap allocation is live at a time instead of the entire file's worth.
pub(crate) fn for_each_packet(data: &[u8], mut f: impl FnMut(TracePacket)) -> std::io::Result<()> {
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
        f(packet);
        pos = end;
    }
    Ok(())
}
