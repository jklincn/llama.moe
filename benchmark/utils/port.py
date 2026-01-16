from __future__ import annotations

from typing import Iterable

import psutil


def kill_process_on_port(
    port: int,
    *,
    timeout_sec: float = 20.0,
    kill_timeout_sec: float = 5.0,
) -> None:
    """Kill processes that are listening on the given TCP port.

    This is intended for local benchmark harness usage.
    """

    def iter_listeners() -> Iterable[psutil._common.sconn]:  # type: ignore[attr-defined]
        try:
            # kind="inet" covers TCP/UDP over IPv4/IPv6.
            return psutil.net_connections(kind="inet")
        except Exception:
            # Fall back to default if kind isn't supported.
            return psutil.net_connections()

    try:
        for conn in iter_listeners():
            if not conn.laddr:
                continue
            if conn.laddr.port != port:
                continue
            if conn.status != psutil.CONN_LISTEN:
                continue
            if conn.pid is None:
                continue

            pid = conn.pid
            try:
                process = psutil.Process(pid)
                print(
                    f"Found process occupying port {port}: PID={pid}, name={process.name()}"
                )

                process.terminate()
                try:
                    process.wait(timeout=timeout_sec)
                    print(
                        f"Successfully terminated process {pid} occupying port {port}"
                    )
                except psutil.TimeoutExpired:
                    print(
                        f"Process {pid} did not exit within {timeout_sec:.0f} seconds, using SIGKILL"
                    )
                    process.kill()
                    process.wait(timeout=kill_timeout_sec)
                    print(f"Process {pid} was forcibly terminated")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Error terminating process {pid}: {e}")
    except Exception as e:
        print(f"Error cleaning up port {port}: {e}")
