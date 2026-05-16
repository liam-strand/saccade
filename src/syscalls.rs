//! Safe wrappers around raw Linux syscalls used for process control, CPU affinity, and scheduling.

use std::io;
use syscalls::{Sysno, syscall4};

/// Calls `ptrace(PTRACE_TRACEME)` so that the calling child process is traced by its parent, causing it to stop on the next `exec`.
pub fn ptrace_traceme() -> io::Result<()> {
    const PTRACE_TRACEME: usize = 0;
    unsafe {
        syscall4(
            Sysno::ptrace,
            PTRACE_TRACEME,
            0, // pid: ignored
            0, // addr: ignored
            0, // data: ignored
        )
        .map_err(|e| io::Error::from_raw_os_error(e.into_raw()))?;
    }
    Ok(())
}

/// Calls `wait4` to block until the child `pid` changes state, returning the raw wait status.
pub fn wait_for_exec(pid: u32) -> io::Result<i32> {
    let mut status: i32 = 0;
    unsafe {
        syscall4(
            Sysno::wait4,
            pid as usize,
            &mut status as *mut i32 as usize,
            0, // No options
            0, // NULL rusage
        )
        .map_err(|e| io::Error::from_raw_os_error(e.into_raw()))?;
    }
    Ok(status)
}

/// Calls `ptrace(PTRACE_DETACH)` to stop tracing `pid` and let it continue execution.
pub fn ptrace_detach(pid: u32) -> io::Result<()> {
    const PTRACE_DETACH: usize = 17;
    unsafe {
        syscall4(
            Sysno::ptrace,
            PTRACE_DETACH,
            pid as usize,
            0, // addr: ignored
            0, // data: signum (0 means no signal)
        )
        .map_err(|e| io::Error::from_raw_os_error(e.into_raw()))?;
    }
    Ok(())
}

/// C-layout equivalent of `cpu_set_t`, holding a 1024-bit CPU bitmask for `sched_setaffinity`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CpuSet {
    /// 1024-bit mask where bit `n` indicates CPU `n` is in the set.
    bits: [u64; 16],
}

impl Default for CpuSet {
    /// Returns an empty `CpuSet` with no CPUs selected.
    fn default() -> Self {
        Self::new()
    }
}

impl CpuSet {
    /// Creates a `CpuSet` with no CPUs selected.
    pub fn new() -> Self {
        Self { bits: [0; 16] }
    }

    /// Sets the bit for `cpu` in the mask; silently ignores indices >= 1024.
    pub fn set(&mut self, cpu: usize) {
        if cpu < 1024 {
            self.bits[cpu / 64] |= 1 << (cpu % 64);
        }
    }
}

/// Calls `sched_setaffinity` to restrict `pid` (or the calling thread when `pid` is 0) to the CPUs in `mask`.
pub fn sched_setaffinity(pid: i32, mask: &CpuSet) -> io::Result<()> {
    unsafe {
        syscalls::syscall3(
            Sysno::sched_setaffinity,
            pid as usize,
            std::mem::size_of::<CpuSet>(),
            mask as *const CpuSet as usize,
        )
        .map_err(|e| io::Error::from_raw_os_error(e.into_raw()))?;
    }
    Ok(())
}

/// Returns the index of the CPU the calling thread is currently running on.
pub fn get_cpu() -> io::Result<usize> {
    let mut cpu: u32 = 0;
    // syscall3(Sysno::getcpu, &mut cpu, NULL, NULL)
    unsafe {
        syscalls::syscall3(Sysno::getcpu, &mut cpu as *mut u32 as usize, 0, 0)
            .map_err(|e| io::Error::from_raw_os_error(e.into_raw()))?;
    }
    Ok(cpu as usize)
}

/// Yields the calling thread's remaining scheduler timeslice.
pub fn sched_yield() -> io::Result<()> {
    unsafe {
        syscalls::syscall0(Sysno::sched_yield)
            .map_err(|e| io::Error::from_raw_os_error(e.into_raw()))?;
    }
    Ok(())
}

/// Returns the thread ID of the calling thread.
pub fn gettid() -> io::Result<usize> {
    unsafe {
        syscalls::syscall0(Sysno::gettid).map_err(|e| io::Error::from_raw_os_error(e.into_raw()))
    }
}
