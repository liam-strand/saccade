//! This module contains safe wrappers around raw syscalls used for process control.
//!
//! We use the `syscalls` crate to invoke Linux system calls directly.
//! This is necessary for fine-grained control over process execution, specifically
//! for `ptrace` operations that are not fully exposed by the standard library.

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
pub fn ptrace_traceme() -> Result<(), syscalls::Errno> {
    const PTRACE_TRACEME: usize = 0;
    unsafe {
        syscall4(
            Sysno::ptrace,
            PTRACE_TRACEME,
            0, // pid: ignored
            0, // addr: ignored
            0, // data: ignored
        )?;
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
pub fn wait_for_exec(pid: u32) -> Result<i32, syscalls::Errno> {
    let mut status: i32 = 0;
    unsafe {
        syscall4(
            Sysno::wait4,
            pid as usize,
            &mut status as *mut i32 as usize,
            0, // No options
            0  // NULL rusage
        )?;
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
pub fn ptrace_detach(pid: u32) -> Result<(), syscalls::Errno> {
    const PTRACE_DETACH: usize = 17;
    unsafe {
        syscall4(
            Sysno::ptrace,
            PTRACE_DETACH,
            pid as usize,
            0, // addr: ignored
            0, // data: signum (0 means no signal)
        )?;
    }
    Ok(())
}
