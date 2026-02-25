use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};

use crate::oculomotor::SaccadeSample;

pub struct Logger {
    sender: Option<SyncSender<SaccadeSample>>,
    handle: Option<JoinHandle<std::io::Result<()>>>,
}

impl Logger {
    pub fn new<P: AsRef<Path>>(path: P, buffer_capacity: usize) -> std::io::Result<Self> {
        let file = File::create(path)?;
        // Use an 8MB buffer for high-performance I/O batching
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

        // Write CSV header
        writeln!(
            writer,
            "timestamp_ns,duration_ns,pid,cpu_id,type,values_0,values_1,values_2,values_3,events_0,events_1,events_2,events_3,task"
        )?;

        let (sender, receiver) = mpsc::sync_channel(buffer_capacity);

        let handle = thread::spawn(move || {
            for sample in receiver {
                write_sample(&mut writer, &sample)?;
            }
            writer.flush()?;
            Ok(())
        });

        Ok(Self {
            sender: Some(sender),
            handle: Some(handle),
        })
    }

    #[inline]
    pub fn log(&self, sample: SaccadeSample) {
        if let Some(sender) = &self.sender {
            // we use send and just block if the queue is full.
            // Alternatively, could use try_send if dropping samples is acceptable
            // over blocking the BPF ringbuffer thread.
            let _ = sender.send(sample);
        }
    }

    pub fn clone_sender(&self) -> Option<SyncSender<SaccadeSample>> {
        self.sender.clone()
    }
}

impl Drop for Logger {
    fn drop(&mut self) {
        // Drop the sender to signal the thread to stop
        self.sender.take();
        // Wait for the thread to finish writing
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn write_sample(writer: &mut impl Write, sample: &SaccadeSample) -> std::io::Result<()> {
    // Fast task name parsing: find null terminator to trim
    let task_len = sample
        .task
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(sample.task.len());
    // Safe fallback to lossy utf-8 in case task name has garbage bytes
    let task_name = String::from_utf8_lossy(&sample.task[..task_len]);

    writeln!(
        writer,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        sample.timestamp_ns,
        sample.duration_ns,
        sample.pid,
        sample.cpu_id,
        sample.type_,
        sample.values[0],
        sample.values[1],
        sample.values[2],
        sample.values[3],
        sample.events[0],
        sample.events[1],
        sample.events[2],
        sample.events[3],
        task_name
    )
}
