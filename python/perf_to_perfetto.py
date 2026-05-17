"""Convert `perf stat -I` CSV output to the Perfetto trace format read by `saccade evaluate`.

Wire format produced here must exactly match what `PerfettoWriter` in `src/perfetto/trace.rs`
writes and what `read_rate_timeseries` in `src/perfetto/reader.rs` reads:

  - The output file is a Perfetto Trace container: a sequence of
    ``0x0A + varint(len) + TracePacket bytes`` records.
  - Counter tracks are named ``{event_name}/rate``; the reader strips this suffix.
  - Counter values use ``double_counter_value`` (events per nanosecond).
  - ``trusted_packet_sequence_id = 1`` on every packet.
  - Because ``perf stat`` aggregates all threads, tid is 0.  The reader defaults
    to tid=0 when a counter track has no parent with a ThreadDescriptor, so we
    emit counter tracks with no parent_uuid at all.

Rate formula (as specified in task):
    rate = (count / running_ns) * (enabled_ns / running_ns)

When running_ns == enabled_ns (no multiplexing) this reduces to count/running_ns.
Under multiplexing this applies a scaling factor of enabled/running, which is the
standard perf scaling approach for the *count* (scaled_count = count * enabled/running),
then divided by running_ns to get a rate. Note this differs from dividing by
enabled_ns; using running_ns as the denominator makes the rate reflect the window
the counter was actually active.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Perfetto proto helpers
# ---------------------------------------------------------------------------

def _write_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    out = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def _write_trace_packet(packet_bytes: bytes) -> bytes:
    """Wrap serialised TracePacket bytes in the Trace container record (tag 0x0A + varint len + data)."""
    return b"\x0A" + _write_varint(len(packet_bytes)) + packet_bytes


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_count(raw: str) -> int | None:
    """Return integer count, or None for perf's not-counted sentinel values."""
    raw = raw.strip()
    if not raw or raw in ("<not counted>", "<not supported>", "<not available>"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_perf_csv(path: Path) -> list[dict]:
    """Parse ``perf stat -I -x,`` CSV output into a list of record dicts.

    Each returned dict has keys:
      timestamp_ns  (int)   – timestamp in nanoseconds from start of run
      event_name    (str)
      count         (int)   – raw hardware count
      running_ns    (int)   – nanoseconds the counter was active
      enabled_ns    (int)   – nanoseconds the counter was requested
      rate          (float) – events per nanosecond, scaled for multiplexing
    """
    records = []
    with path.open(newline="") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            # csv.reader handles quoting; fields may be unquoted
            row = next(csv.reader([line]))
            # Pad to at least 7 fields to tolerate short/trailing rows
            while len(row) < 7:
                row.append("")

            timestamp_s_str = row[0].strip()
            count_str = row[1].strip()
            # row[2] = unit (ignored)
            event_name = row[3].strip()
            running_ns_str = row[4].strip()
            enabled_ns_str = row[5].strip()
            # row[6] = utilization_pct (ignored; we compute our own)

            # Parse timestamp
            try:
                timestamp_ns = int(float(timestamp_s_str) * 1e9)
            except ValueError:
                continue

            # Parse count
            count = _parse_count(count_str)
            if count is None:
                continue

            # Parse running / enabled nanoseconds
            try:
                running_ns = int(running_ns_str) if running_ns_str else 0
                enabled_ns = int(enabled_ns_str) if enabled_ns_str else 0
            except ValueError:
                running_ns = 0
                enabled_ns = 0

            if running_ns <= 0:
                rate = 0.0
            else:
                # Standard perf scaling: rate in events/ns.
                # rate = (count / running_ns) * (enabled_ns / running_ns)
                rate = (count / running_ns) * (enabled_ns / running_ns)

            if not event_name:
                continue

            records.append(
                {
                    "timestamp_ns": timestamp_ns,
                    "event_name": event_name,
                    "count": count,
                    "running_ns": running_ns,
                    "enabled_ns": enabled_ns,
                    "rate": rate,
                }
            )

    return records


# ---------------------------------------------------------------------------
# Perfetto writing
# ---------------------------------------------------------------------------

def _build_track_descriptor_packet(
    uuid: int,
    name: str,
    *,
    parent_uuid: int | None = None,
    thread_pid: int | None = None,
    thread_tid: int | None = None,
    thread_name: str | None = None,
    is_counter: bool = False,
    counter_unit_name: str | None = None,
) -> bytes:
    """Build and return the raw bytes of a TracePacket wrapping a TrackDescriptor."""
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TracePacket

    pkt = TracePacket()
    pkt.trusted_packet_sequence_id = 1

    td = pkt.track_descriptor
    td.uuid = uuid
    td.name = name
    if parent_uuid is not None:
        td.parent_uuid = parent_uuid

    if thread_pid is not None and thread_tid is not None:
        th = td.thread
        th.pid = thread_pid
        th.tid = thread_tid
        if thread_name:
            th.thread_name = thread_name

    if is_counter:
        ctr = td.counter
        if counter_unit_name:
            ctr.unit_name = counter_unit_name

    return pkt.SerializeToString()


def _build_counter_packet(timestamp_ns: int, track_uuid: int, value: float) -> bytes:
    """Build and return the raw bytes of a TYPE_COUNTER TracePacket."""
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TracePacket

    pkt = TracePacket()
    pkt.trusted_packet_sequence_id = 1
    pkt.timestamp = timestamp_ns

    te = pkt.track_event
    te.type = 4  # TYPE_COUNTER
    te.track_uuid = track_uuid
    te.double_counter_value = value

    return pkt.SerializeToString()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert(input_path: Path, output_path: Path, interval_ms: float = 100.0) -> None:
    """Convert a ``perf stat -I`` CSV file at *input_path* to a Perfetto trace at *output_path*.

    The interval_ms parameter is accepted for API compatibility but is not used
    in the rate computation; rates are derived directly from the running_ns and
    enabled_ns columns reported by perf stat.

    The output format matches what ``saccade evaluate`` expects:
    - One counter track per event, named ``{event_name}/rate``.
    - Counter values in events per nanosecond.
    - tid defaults to 0 (perf stat aggregates all threads).
    """
    records = parse_perf_csv(input_path)
    if not records:
        raise ValueError(f"No valid records found in {input_path}")

    # Collect unique event names and assign UUIDs starting at 1_000_000
    # (mirrors THREAD_UUID_BASE in the Rust writer).
    event_names: list[str] = []
    seen_events: set[str] = set()
    for rec in records:
        name = rec["event_name"]
        if name not in seen_events:
            seen_events.add(name)
            event_names.append(name)

    uuid_base = 1_000_000
    event_uuid: dict[str, int] = {
        name: uuid_base + i for i, name in enumerate(event_names)
    }

    with output_path.open("wb") as fh:
        # Emit one TrackDescriptor per event (counter track, no parent → tid=0).
        for name in event_names:
            pkt_bytes = _build_track_descriptor_packet(
                uuid=event_uuid[name],
                name=f"{name}/rate",
                is_counter=True,
                counter_unit_name="events/ns",
            )
            fh.write(_write_trace_packet(pkt_bytes))

        # Emit counter values in timestamp order (already sorted by perf stat).
        for rec in records:
            name = rec["event_name"]
            uuid = event_uuid[name]
            pkt_bytes = _build_counter_packet(rec["timestamp_ns"], uuid, rec["rate"])
            fh.write(_write_trace_packet(pkt_bytes))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert perf stat -I CSV output to a Perfetto trace file "
            "compatible with 'saccade evaluate'."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to the perf stat CSV file (produced by: perf stat -I <ms> -x, ...).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path for the output Perfetto trace file.",
    )
    p.add_argument(
        "--interval-ms",
        type=float,
        default=100.0,
        metavar="MS",
        help=(
            "The -I interval (in milliseconds) used with perf stat. "
            "Accepted for compatibility; rates are derived from perf's own "
            "running_ns / enabled_ns columns. Default: 100."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    convert(args.input, args.output, args.interval_ms)
    print(f"Written: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
