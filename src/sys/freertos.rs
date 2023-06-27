// Copyright 2015 The Rust Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::min;
use std::io::IoSlice;
use std::marker::PhantomData;
use std::mem::forget;
use std::mem::{self, size_of, MaybeUninit};
use std::net::Shutdown;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::os::freertos::io::RawSocket;
use std::os::freertos::io::{AsRawSocket, FromRawSocket, IntoRawSocket};
use std::ptr;
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{io, slice};

use lwip as netc;

use crate::RecvFlags;
use crate::{Domain, Protocol, SockAddr, TcpKeepalive, Type};

pub(crate) use core::ffi::{c_int, c_long, c_void};

// Used in `Domain`.
pub(crate) use netc::{AF_INET, AF_INET6};
// Used in `Type`.
#[cfg(feature = "all")]
pub(crate) use netc::SOCK_RAW;
pub(crate) use netc::{SOCK_DGRAM, SOCK_STREAM};

// Used in `Protocol`.
pub(crate) use netc::{IPPROTO_ICMP, IPPROTO_ICMPV6, IPPROTO_TCP, IPPROTO_UDP};
// Used in `SockAddr`.
pub(crate) use netc::{
    sa_family_t, sockaddr, sockaddr_in, sockaddr_in6, sockaddr_storage, socklen_t,
};
// Used in `RecvFlags`.
pub(crate) use netc::{MSG_TRUNC, SO_OOBINLINE};
// Used in `Socket`.
pub(crate) use netc::FIONBIO;

pub(crate) use netc::IP_TOS;
pub(crate) use netc::SO_LINGER;
pub(crate) use netc::{
    ip_mreq as IpMreq, linger, IPPROTO_IP, IPPROTO_IPV6, IPV6_V6ONLY, IP_ADD_MEMBERSHIP,
    IP_DROP_MEMBERSHIP, IP_MULTICAST_IF, IP_MULTICAST_LOOP, IP_MULTICAST_TTL, IP_TTL, MSG_OOB,
    MSG_PEEK, SOL_SOCKET, SO_BROADCAST, SO_ERROR, SO_KEEPALIVE, SO_RCVBUF, SO_RCVTIMEO,
    SO_REUSEADDR, SO_SNDBUF, SO_SNDTIMEO, SO_TYPE, TCP_NODELAY,
};

#[cfg(all(feature = "all"))]
pub(crate) use netc::{TCP_KEEPCNT, TCP_KEEPINTVL};

// See this type in the Windows file.
pub(crate) type Bool = c_int;

use netc::TCP_KEEPIDLE as KEEPALIVE_TIME;

pub fn wait_for_lwip_init() {
    // The LwIP initialisation function can only be called once, and should be called at startup (as part of the OS
    // initialisation). So, no need to call it here. Instead, we can check whether initialisation is complete, and wait for it.
    // We wait a limited time so that network functions don't block indefinitely. If initialisation hasn't finished by then,
    // the network function will fail with an error message.
    let mut retry_count = 0;
    loop {
        if netc::is_netif_initialised() {
            return;
        }
        if retry_count > 12 {
            return;
        }
        sleep(Duration::from_millis(250));
        retry_count = retry_count + 1;
    }
}

/// Helper macro to execute a system call that returns an `io::Result`.
macro_rules! syscall {
    ($fn: ident ( $($arg: expr),* $(,)* ) ) => {{
        #[allow(unused_unsafe)]
        wait_for_lwip_init();
        let res = netc::$fn($($arg, )*);
        if res == -1 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(res)
        }
    }};
}

/// Maximum size of a buffer passed to system call like `recv` and `send`.
const MAX_BUF_LEN: usize = <c_int>::max_value() as usize;

type IovLen = c_int;

// FreeRTOS only API: No specific 'Domain' features.

impl_debug!(Domain, netc::AF_INET, netc::AF_INET6,);

// FreeRTOS only API: No specific 'Type' features.

impl_debug!(Type, netc::SOCK_STREAM, netc::SOCK_DGRAM, netc::SOCK_RAW,);

impl_debug!(
    Protocol,
    netc::IPPROTO_ICMP,
    netc::IPPROTO_ICMPV6,
    netc::IPPROTO_TCP,
    netc::IPPROTO_UDP,
);

// FreeRTOS only API: No specific 'RecvFlags' features.

impl std::fmt::Debug for RecvFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecvFlags")
            .field("is_truncated", &self.is_truncated())
            .finish()
    }
}

#[repr(transparent)]
pub struct MaybeUninitSlice<'a> {
    vec: netc::iovec,
    _lifetime: PhantomData<&'a mut [MaybeUninit<u8>]>,
}

unsafe impl<'a> Send for MaybeUninitSlice<'a> {}

unsafe impl<'a> Sync for MaybeUninitSlice<'a> {}

impl<'a> MaybeUninitSlice<'a> {
    pub(crate) fn new(buf: &'a mut [MaybeUninit<u8>]) -> MaybeUninitSlice<'a> {
        MaybeUninitSlice {
            vec: netc::iovec {
                iov_base: buf.as_mut_ptr().cast(),
                iov_len: buf.len() as i32,
            },
            _lifetime: PhantomData,
        }
    }

    pub(crate) fn as_slice(&self) -> &[MaybeUninit<u8>] {
        unsafe { slice::from_raw_parts(self.vec.iov_base.cast(), self.vec.iov_len as usize) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [MaybeUninit<u8>] {
        unsafe { slice::from_raw_parts_mut(self.vec.iov_base.cast(), self.vec.iov_len as usize) }
    }
}

/// FreeRTOS only API: No specific 'SockAddr' features.

//pub(crate) type Socket = c_int;
pub(crate) type Socket = RawSocket;

pub(crate) unsafe fn socket_from_raw(socket: Socket) -> crate::socket::Inner {
    crate::socket::Inner::from_raw_socket(socket)
}

pub(crate) fn socket_as_raw(socket: &crate::socket::Inner) -> Socket {
    (*socket).as_raw_socket()
}

pub(crate) fn socket_into_raw(socket: crate::socket::Inner) -> Socket {
    let raw_socket = socket.as_raw_socket() as Socket;
    forget(socket);
    raw_socket
}

pub(crate) fn socket(family: c_int, ty: c_int, protocol: c_int) -> io::Result<Socket> {
    syscall!(socket(family, ty, protocol))
}

// socketpair not supported

pub(crate) fn bind(fd: Socket, addr: &SockAddr) -> io::Result<()> {
    syscall!(bind(fd, addr.as_ptr(), addr.len() as _)).map(|_| ())
}

pub(crate) fn connect(fd: Socket, addr: &SockAddr) -> io::Result<()> {
    syscall!(connect(fd, addr.as_ptr(), addr.len())).map(|_| ())
}

pub(crate) fn poll_connect(socket: &crate::Socket, timeout: Duration) -> io::Result<()> {
    let start = Instant::now();

    let mut pollfd = netc::pollfd {
        fd: socket.as_raw(),
        events: netc::POLLIN | netc::POLLOUT,
        revents: 0,
    };

    loop {
        let elapsed = start.elapsed();
        if elapsed >= timeout {
            return Err(io::ErrorKind::TimedOut.into());
        }

        let timeout = (timeout - elapsed).as_millis();
        let timeout = clamp(timeout, 1, c_int::max_value() as u128) as c_int;

        match syscall!(poll(&mut pollfd, 1, timeout)) {
            Ok(0) => return Err(io::ErrorKind::TimedOut.into()),
            Ok(_) => {
                // Error or hang up indicates an error (or failure to connect).
                if (pollfd.revents & netc::POLLHUP) != 0 || (pollfd.revents & netc::POLLERR) != 0 {
                    match socket.take_error() {
                        Ok(Some(err)) => return Err(err),
                        Ok(None) => {
                            return Err(io::Error::new(
                                io::ErrorKind::Other,
                                "no error set after POLLHUP",
                            ))
                        }
                        Err(err) => return Err(err),
                    }
                }
                return Ok(());
            }
            // Got interrupted, try again.
            Err(ref err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
}

// TODO: use clamp from std lib, stable since 1.50.
fn clamp<T>(value: T, min: T, max: T) -> T
where
    T: Ord,
{
    if value <= min {
        min
    } else if value >= max {
        max
    } else {
        value
    }
}

pub(crate) fn listen(fd: Socket, backlog: c_int) -> io::Result<()> {
    syscall!(listen(fd, backlog)).map(|_| ())
}

pub(crate) fn accept(fd: Socket) -> io::Result<(Socket, SockAddr)> {
    // Safety: `accept` initialises the `SockAddr` for us.
    unsafe { SockAddr::init(|storage, len| syscall!(accept(fd, storage.cast(), len))) }
}

pub(crate) fn getsockname(fd: Socket) -> io::Result<SockAddr> {
    unsafe { SockAddr::init(|storage, len| syscall!(getsockname(fd, storage.cast(), len))) }
        .map(|(_, addr)| addr)
}

pub(crate) fn getpeername(fd: Socket) -> io::Result<SockAddr> {
    unsafe { SockAddr::init(|storage, len| syscall!(getpeername(fd, storage.cast(), len))) }
        .map(|(_, addr)| addr)
}

// try_clone: We could just clone the socket, but there are hazards with implicit closing of socket on drop.
// There is no support in LwIP for managed copies (fcntl does not support it, for sure)
// Disabled for now - may need to come back to this if needed
/*
pub(crate) fn try_clone(fd: Socket) -> io::Result<Socket> {
    syscall!(fcntl(fd, libc::F_DUPFD_CLOEXEC, 0))
}
*/

pub(crate) fn set_nonblocking(fd: Socket, nonblocking: bool) -> io::Result<()> {
    // Generic lwip_ioctl function takes a mutable argument pointer.  In this case (FIONBIO), the argument is not written.
    // To keep Rust happy, we need to make a mutable copy to pass to the function
    // Furthermore, we need to pass lwip_ioctl a pointer to int - lwip_ioctl evaluates 4 bytes (nonzero = false).
    // So, an 8-bit false bool alongside nonzero bytes will be evaluated as true!
    let mut nonblocking_mut = nonblocking as c_int;

    syscall!(ioctl(
        fd,
        FIONBIO,
        &mut nonblocking_mut as *mut _ as *mut c_void,
    ))
    .map(|_| ())
}

pub(crate) fn shutdown(fd: Socket, how: Shutdown) -> io::Result<()> {
    let how = match how {
        Shutdown::Write => netc::SHUT_WR,
        Shutdown::Read => netc::SHUT_RD,
        Shutdown::Both => netc::SHUT_RDWR,
    };
    syscall!(shutdown(fd, how)).map(|_| ())
}

pub(crate) fn recv(fd: Socket, buf: &mut [MaybeUninit<u8>], flags: c_int) -> io::Result<usize> {
    syscall!(recv(
        fd,
        buf.as_mut_ptr().cast(),
        min(buf.len(), MAX_BUF_LEN) as c_int,
        flags,
    ))
    .map(|n| n as usize)
}

pub(crate) fn recv_from(
    fd: Socket,
    buf: &mut [MaybeUninit<u8>],
    flags: c_int,
) -> io::Result<(usize, SockAddr)> {
    // Safety: `recvfrom` initialises the `SockAddr` for us.
    unsafe {
        SockAddr::init(|addr, addrlen| {
            syscall!(recvfrom(
                fd,
                buf.as_mut_ptr().cast(),
                min(buf.len(), MAX_BUF_LEN) as c_int,
                flags,
                addr.cast(),
                addrlen
            ))
            .map(|n| n as usize)
        })
    }
}

pub(crate) fn peek_sender(fd: Socket) -> io::Result<SockAddr> {
    // Unix-like platforms simply truncate the returned data, so this implementation is trivial.
    // However, for Windows this requires suppressing the `WSAEMSGSIZE` error,
    // so that requires a different approach.
    // NOTE: macOS does not populate `sockaddr` if you pass a zero-sized buffer.
    let (_, sender) = recv_from(fd, &mut [MaybeUninit::uninit(); 8], MSG_PEEK)?;
    Ok(sender)
}

pub(crate) fn recv_vectored(
    fd: Socket,
    bufs: &mut [crate::MaybeUninitSlice<'_>],
    flags: c_int,
) -> io::Result<(usize, RecvFlags)> {
    recvmsg(fd, ptr::null_mut(), bufs, flags).map(|(n, _, recv_flags)| (n, recv_flags))
}

pub(crate) fn recv_from_vectored(
    fd: Socket,
    bufs: &mut [crate::MaybeUninitSlice<'_>],
    flags: c_int,
) -> io::Result<(usize, RecvFlags, SockAddr)> {
    // Safety: `recvmsg` initialises the address storage and we set the length
    // manually.
    unsafe {
        SockAddr::init(|storage, len| {
            recvmsg(fd, storage, bufs, flags).map(|(n, addrlen, recv_flags)| {
                // Set the correct address length.
                *len = addrlen;
                (n, recv_flags)
            })
        })
    }
    .map(|((n, recv_flags), addr)| (n, recv_flags, addr))
}

/// Returns the (bytes received, sending address len, `RecvFlags`).
fn recvmsg(
    fd: Socket,
    msg_name: *mut sockaddr_storage,
    bufs: &mut [crate::MaybeUninitSlice<'_>],
    flags: c_int,
) -> io::Result<(usize, netc::socklen_t, RecvFlags)> {
    let msg_namelen = if msg_name.is_null() {
        0
    } else {
        size_of::<sockaddr_storage>() as netc::socklen_t
    };

    let mut msg: netc::msghdr = unsafe { mem::zeroed() };
    msg.msg_name = msg_name.cast();
    msg.msg_namelen = msg_namelen;
    msg.msg_iov = bufs.as_mut_ptr().cast();
    msg.msg_iovlen = min(bufs.len(), IovLen::MAX as usize) as IovLen;
    syscall!(recvmsg(fd, &mut msg, flags))
        .map(|n| (n as usize, msg.msg_namelen, RecvFlags(msg.msg_flags)))
}

pub(crate) fn send(fd: Socket, buf: &[u8], flags: c_int) -> io::Result<usize> {
    syscall!(send(
        fd,
        buf.as_ptr().cast(),
        min(buf.len(), MAX_BUF_LEN) as c_int,
        flags,
    ))
    .map(|n| n as usize)
}

pub(crate) fn send_vectored(fd: Socket, bufs: &[IoSlice<'_>], flags: c_int) -> io::Result<usize> {
    sendmsg(fd, ptr::null(), 0, bufs, flags)
}

pub(crate) fn send_to(fd: Socket, buf: &[u8], addr: &SockAddr, flags: c_int) -> io::Result<usize> {
    syscall!(sendto(
        fd,
        buf.as_ptr().cast(),
        min(buf.len(), MAX_BUF_LEN) as c_int,
        flags,
        addr.as_ptr(),
        addr.len(),
    ))
    .map(|n| n as usize)
}

pub(crate) fn send_to_vectored(
    fd: Socket,
    bufs: &[IoSlice<'_>],
    addr: &SockAddr,
    flags: c_int,
) -> io::Result<usize> {
    sendmsg(fd, addr.as_storage_ptr(), addr.len(), bufs, flags)
}

/// Returns the (bytes received, sending address len, `RecvFlags`).
fn sendmsg(
    fd: Socket,
    msg_name: *const sockaddr_storage,
    msg_namelen: socklen_t,
    bufs: &[IoSlice<'_>],
    flags: c_int,
) -> io::Result<usize> {
    let mut msg: netc::msghdr = unsafe { mem::zeroed() };
    // Safety: we're creating a `*mut` pointer from a reference, which is UB
    // once actually used. However the OS should not write to it in the
    // `sendmsg` system call.
    msg.msg_name = (msg_name as *mut sockaddr_storage).cast();
    msg.msg_namelen = msg_namelen;
    // Safety: Same as above about `*const` -> `*mut`.
    msg.msg_iov = bufs.as_ptr() as *mut _;
    msg.msg_iovlen = min(bufs.len(), IovLen::MAX as usize) as IovLen;
    syscall!(sendmsg(fd, &msg, flags)).map(|n| n as usize)
}

/// Wrapper around `getsockopt` to deal with platform specific timeouts.
pub(crate) fn timeout_opt(fd: Socket, opt: c_int, val: c_int) -> io::Result<Option<Duration>> {
    unsafe { getsockopt(fd, opt, val).map(from_timeval) }
}

fn from_timeval(duration: netc::timeval) -> Option<Duration> {
    if duration.tv_sec == 0 && duration.tv_usec == 0 {
        None
    } else {
        let sec = duration.tv_sec as u64;
        let nsec = (duration.tv_usec as u32) * 1000;
        Some(Duration::new(sec, nsec))
    }
}

/// Wrapper around `setsockopt` to deal with platform specific timeouts.
pub(crate) fn set_timeout_opt(
    fd: Socket,
    opt: c_int,
    val: c_int,
    duration: Option<Duration>,
) -> io::Result<()> {
    let duration = into_timeval(duration);
    unsafe { setsockopt(fd, opt, val, duration) }
}

fn into_timeval(duration: Option<Duration>) -> netc::timeval {
    match duration {
        // https://github.com/rust-lang/libc/issues/1848
        #[cfg_attr(target_env = "musl", allow(deprecated))]
        Some(duration) => netc::timeval {
            tv_sec: duration.as_secs() as i64,
            tv_usec: duration.subsec_micros() as c_long,
        },
        None => netc::timeval {
            tv_sec: 0,
            tv_usec: 0,
        },
    }
}

#[cfg(feature = "all")]
pub(crate) fn keepalive_time(fd: Socket) -> io::Result<Duration> {
    unsafe {
        getsockopt::<c_int>(fd, IPPROTO_TCP, KEEPALIVE_TIME)
            .map(|secs| Duration::from_secs(secs as u64))
    }
}

#[allow(unused_variables)]
pub(crate) fn set_tcp_keepalive(fd: Socket, keepalive: &TcpKeepalive) -> io::Result<()> {
    if let Some(time) = keepalive.time {
        let secs = into_secs(time);
        unsafe { setsockopt(fd, netc::IPPROTO_TCP, KEEPALIVE_TIME, secs)? }
    }

    {
        if let Some(interval) = keepalive.interval {
            let secs = into_secs(interval);
            unsafe { setsockopt(fd, netc::IPPROTO_TCP, TCP_KEEPINTVL, secs)? }
        }

        if let Some(retries) = keepalive.retries {
            unsafe { setsockopt(fd, netc::IPPROTO_TCP, TCP_KEEPCNT, retries as c_int)? }
        }
    }

    Ok(())
}

fn into_secs(duration: Duration) -> c_int {
    min(duration.as_secs(), c_int::max_value() as u64) as c_int
}

/*
/// Add `flag` to the current set flags of `F_GETFD`.
fn fcntl_add(fd: Socket, get_cmd: c_int, set_cmd: c_int, flag: c_int) -> io::Result<()> {
    let previous = syscall!(fcntl(fd, get_cmd))?;
    let new = previous | flag;
    if new != previous {
        syscall!(fcntl(fd, set_cmd, new)).map(|_| ())
    } else {
        // Flag was already set.
        Ok(())
    }
}

/// Remove `flag` to the current set flags of `F_GETFD`.
fn fcntl_remove(fd: Socket, get_cmd: c_int, set_cmd: c_int, flag: c_int) -> io::Result<()> {
    let previous = syscall!(fcntl(fd, get_cmd))?;
    let new = previous & !flag;
    if new != previous {
        syscall!(fcntl(fd, set_cmd, new)).map(|_| ())
    } else {
        // Flag was already set.
        Ok(())
    }
}
*/

/// Caller must ensure `T` is the correct type for `opt` and `val`.
pub(crate) unsafe fn getsockopt<T>(fd: Socket, opt: c_int, val: c_int) -> io::Result<T> {
    let mut payload: MaybeUninit<T> = MaybeUninit::uninit();
    let mut len = size_of::<T>() as netc::socklen_t;
    syscall!(getsockopt(
        fd,
        opt,
        val,
        payload.as_mut_ptr().cast(),
        &mut len,
    ))
    .map(|_| {
        debug_assert_eq!(len as usize, size_of::<T>());
        // Safety: `getsockopt` initialised `payload` for us.
        payload.assume_init()
    })
}

/// Caller must ensure `T` is the correct type for `opt` and `val`.
pub(crate) unsafe fn setsockopt<T>(
    fd: Socket,
    opt: c_int,
    val: c_int,
    payload: T,
) -> io::Result<()> {
    let payload = &payload as *const T as *const c_void;
    syscall!(setsockopt(
        fd,
        opt,
        val,
        payload,
        mem::size_of::<T>() as netc::socklen_t,
    ))
    .map(|_| ())
}

pub(crate) fn to_in_addr(addr: &Ipv4Addr) -> netc::in_addr {
    // `s_addr` is stored as BE on all machines, and the array is in BE order.
    // So the native endian conversion method is used so that it's never
    // swapped.
    netc::in_addr {
        s_addr: u32::from_ne_bytes(addr.octets()),
    }
}

pub(crate) fn from_in_addr(in_addr: netc::in_addr) -> Ipv4Addr {
    Ipv4Addr::from(in_addr.s_addr.to_ne_bytes())
}

pub(crate) fn to_in6_addr(addr: &Ipv6Addr) -> netc::in6_addr {
    netc::in6_addr {
        s6_addr: addr.octets(),
    }
}

pub(crate) fn from_in6_addr(addr: netc::in6_addr) -> Ipv6Addr {
    Ipv6Addr::from(addr.s6_addr)
}

/// FreeRTOS only API
impl crate::Socket {
    /// Returns `true` if `listen(2)` was called on this socket by checking the
    /// `SO_ACCEPTCONN` option on this socket.
    #[cfg(feature = "all")]
    #[cfg_attr(docsrs, doc(cfg(feature = "all")))]
    pub fn is_listener(&self) -> io::Result<bool> {
        unsafe {
            getsockopt::<c_int>(self.as_raw(), netc::SOL_SOCKET, netc::SO_ACCEPTCONN)
                .map(|v| v != 0)
        }
    }

    /// Gets the value for the `SO_BINDTODEVICE` option on this socket.
    ///
    /// This value gets the socket binded device's interface name.
    #[cfg(feature = "all")]
    #[cfg_attr(docsrs, doc(cfg(feature = "all")))]
    pub fn device(&self) -> io::Result<Option<Vec<u8>>> {
        // TODO: replace with `MaybeUninit::uninit_array` once stable.
        // TODO: link array size to a define in LwIP (if one exists)
        let mut buf: [MaybeUninit<u8>; 8] = unsafe { MaybeUninit::uninit().assume_init() };
        let mut len = buf.len() as netc::socklen_t;
        syscall!(getsockopt(
            self.as_raw(),
            netc::SOL_SOCKET,
            netc::SO_BINDTODEVICE,
            buf.as_mut_ptr().cast(),
            &mut len,
        ))?;
        if len == 0 {
            Ok(None)
        } else {
            let buf = &buf[..len as usize - 1];
            // TODO: use `MaybeUninit::slice_assume_init_ref` once stable.
            Ok(Some(unsafe { &*(buf as *const [_] as *const [u8]) }.into()))
        }
    }

    /// Sets the value for the `SO_BINDTODEVICE` option on this socket.
    ///
    /// If a socket is bound to an interface, only packets received from that
    /// particular interface are processed by the socket. Note that this only
    /// works for some socket types, particularly `AF_INET` sockets.
    ///
    /// If `interface` is `None` or an empty string it removes the binding.
    #[cfg(feature = "all")]
    #[cfg_attr(docsrs, doc(cfg(feature = "all")))]
    pub fn bind_device(&self, interface: Option<&[u8]>) -> io::Result<()> {
        let (value, len) = if let Some(interface) = interface {
            (interface.as_ptr(), interface.len())
        } else {
            (ptr::null(), 0)
        };
        syscall!(setsockopt(
            self.as_raw(),
            netc::SOL_SOCKET,
            netc::SO_BINDTODEVICE,
            value.cast(),
            len as netc::socklen_t,
        ))
        .map(|_| ())
    }

    /// Get the value of the `SO_REUSEPORT` option on this socket.
    ///
    /// For more information about this option, see [`set_reuse_port`].
    ///
    /// [`set_reuse_port`]: crate::Socket::set_reuse_port
    #[cfg(feature = "all")]
    #[cfg_attr(docsrs, doc(cfg(feature = "all")))]
    pub fn reuse_port(&self) -> io::Result<bool> {
        unsafe {
            getsockopt::<c_int>(self.as_raw(), netc::SOL_SOCKET, netc::SO_REUSEPORT)
                .map(|reuse| reuse != 0)
        }
    }

    /// Set value for the `SO_REUSEPORT` option on this socket.
    ///
    /// This indicates that further calls to `bind` may allow reuse of local
    /// addresses. For IPv4 sockets this means that a socket may bind even when
    /// there's a socket already listening on this port.
    #[cfg(feature = "all")]
    #[cfg_attr(docsrs, doc(cfg(feature = "all")))]
    pub fn set_reuse_port(&self, reuse: bool) -> io::Result<()> {
        unsafe {
            setsockopt(
                self.as_raw(),
                netc::SOL_SOCKET,
                netc::SO_REUSEPORT,
                reuse as c_int,
            )
        }
    }
}

impl AsRawSocket for crate::Socket {
    fn as_raw_socket(&self) -> RawSocket {
        self.as_raw() as RawSocket
    }
}

impl IntoRawSocket for crate::Socket {
    fn into_raw_socket(self) -> RawSocket {
        self.into_raw() as RawSocket
    }
}

impl FromRawSocket for crate::Socket {
    unsafe fn from_raw_socket(socket: RawSocket) -> crate::Socket {
        crate::Socket::from_raw(socket as Socket)
    }
}

#[test]
fn in_addr_convertion() {
    let ip = Ipv4Addr::new(127, 0, 0, 1);
    let raw = to_in_addr(&ip);
    // NOTE: `in_addr` is packed on NetBSD and it's unsafe to borrow.
    let a = raw.s_addr;
    assert_eq!(a, u32::from_ne_bytes([127, 0, 0, 1]));
    assert_eq!(from_in_addr(raw), ip);

    let ip = Ipv4Addr::new(127, 34, 4, 12);
    let raw = to_in_addr(&ip);
    let a = raw.s_addr;
    assert_eq!(a, u32::from_ne_bytes([127, 34, 4, 12]));
    assert_eq!(from_in_addr(raw), ip);
}

#[test]
fn in6_addr_convertion() {
    let ip = Ipv6Addr::new(0x2000, 1, 2, 3, 4, 5, 6, 7);
    let raw = to_in6_addr(&ip);
    let want = [32, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7];
    assert_eq!(raw.s6_addr, want);
    assert_eq!(from_in6_addr(raw), ip);
}
