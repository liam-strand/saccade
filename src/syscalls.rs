//! This module contains safe wrappers around raw syscalls used for process control.
//!
//! We use the `syscalls` crate to invoke Linux system calls directly.
//! This is necessary for fine-grained control over process execution, specifically
//! for `ptrace` operations that are not fully exposed by the standard library.

use std::io;
use syscalls::{Sysno, syscall4};

/// Invokes the `ptrace` syscall with `PTRACE_TRACEME`.
///
/// This indicates that this process is to be traced by its parent.
/// It is typically called by a child process immediately after `fork()` and before `exec()`.
///
/// When a process calls this, it will be stopped (via `SIGTRAP`) by the kernel
/// upon the next successful `exec()` call. This allows the parent to gain control
/// at the very beginning of the new program's execution.
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if the syscall fails.
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

/// Invokes the `wait4` syscall to wait for a specific child process to change state.
///
/// This function is specifically designed to wait for the child process to stop
/// after it has called `exec()` (which it does because of `PTRACE_TRACEME`).
///
/// # Arguments
///
/// * `pid` - The process ID of the child to wait for.
///
/// # Returns
///
/// Returns `Ok(status)` containing the wait status, or an error if the syscall fails.
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

/// Invokes the `ptrace` syscall with `PTRACE_DETACH`.
///
/// This detaches the tracer (parent) from the tracee (child) and allows the child
/// to continue execution. It effectively "resumes" the child process.
///
/// # Arguments
///
/// * `pid` - The process ID of the child to detach from.
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if the syscall fails.
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

/// A minimal wrapper around cpu_set_t for sched_setaffinity.
/// Linux kernel expects a bitmask.
/// We implement a fixed size set (1024 bits = 128 bytes) which is standard.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CpuSet {
    bits: [u64; 16], // 16 * 64 = 1024 bits
}

impl CpuSet {
    pub fn new() -> Self {
        Self { bits: [0; 16] }
    }

    pub fn set(&mut self, cpu: usize) {
        if cpu < 1024 {
            self.bits[cpu / 64] |= 1 << (cpu % 64);
        }
    }
}

/// Invokes the `sched_setaffinity` syscall to pin the current process/thread to specific CPUs.
///
/// # Arguments
///
/// * `pid` - The process ID to set affinity for. 0 means current thread.
/// * `mask` - The CPU set bitmask.
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if the syscall fails.
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

/// Helper to get the current CPU index using the getcpu syscall.
pub fn get_cpu() -> io::Result<usize> {
    let mut cpu: u32 = 0;
    // syscall3(Sysno::getcpu, &mut cpu, NULL, NULL)
    unsafe {
        syscalls::syscall3(Sysno::getcpu, &mut cpu as *mut u32 as usize, 0, 0)
            .map_err(|e| io::Error::from_raw_os_error(e.into_raw()))?;
    }
    Ok(cpu as usize)
}

/// Invokes the `sched_yield` syscall.
pub fn sched_yield() -> io::Result<()> {
    unsafe {
        syscalls::syscall0(Sysno::sched_yield)
            .map_err(|e| io::Error::from_raw_os_error(e.into_raw()))?;
    }
    Ok(())
}
