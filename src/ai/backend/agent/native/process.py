"""On-disk record and signalling of a kernel that runs as a host process tree."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import socket
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath
from typing import Final

import psutil
from pydantic import BaseModel

from ai.backend.agent.types import Container, Port
from ai.backend.common.types import ContainerId, ContainerStatus, KernelId

KERNEL_HOME: Final = PurePosixPath("/home/work")
PROCESS_INFO_FILENAME: Final = "native-kernel.json"
LOG_FILENAME: Final = "kernel.log"
_CREATE_TIME_TOLERANCE: Final = 1.0


class KernelProcessInfo(BaseModel):
    """What the agent needs to find, signal and account a kernel without its in-memory object."""

    container_id: str
    pid: int  # also the process group id, since the runner leads its own session
    create_time: float  # guards against pid reuse
    image: str
    labels: dict[str, str]
    host_ports: list[int]
    service_ports: list[int] = []  # declared by the kernel; they are host ports as they are
    # Set once the agent ended the kernel or reported its death; a clean without it is a death.
    exit_handled: bool = False

    def _is_leader(self, proc: psutil.Process) -> bool:
        return abs(proc.create_time() - self.create_time) < _CREATE_TIME_TOLERANCE

    def is_alive(self) -> bool:
        try:
            proc = psutil.Process(self.pid)
            return self._is_leader(proc) and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    def owns_group(self) -> bool:
        """False when the pid now belongs to an unrelated process, so the pgid may too."""
        try:
            return self._is_leader(psutil.Process(self.pid))
        except psutil.NoSuchProcess:
            # A pgid is not reused while members remain, so leftovers are still ours.
            return True
        except psutil.Error:
            return False

    def to_container(self, config_dir: Path) -> Container:
        return Container(
            id=ContainerId(self.container_id),
            status=ContainerStatus.RUNNING if self.is_alive() else ContainerStatus.EXITED,
            image=self.image,
            labels=self.labels,
            ports=[Port("127.0.0.1", port, port) for port in self.host_ports],
            # Docker-shaped so that compute plugins find resource.txt the same way.
            backend_obj={
                "HostConfig": {"Mounts": [{"Source": str(config_dir), "Target": "/home/config"}]},
            },
        )


def config_dir_of(scratch_root: Path, kernel_id: KernelId) -> Path:
    return (scratch_root / str(kernel_id)).resolve() / "config"


def host_path_of(scratch_dir: Path, kernel_path: str | os.PathLike[str]) -> Path:
    """Map a kernel-side absolute path: /home/work/x to work/x, any other /x to mounts/x."""
    target = PurePosixPath(os.path.normpath(kernel_path))
    if target.is_relative_to(KERNEL_HOME):
        return scratch_dir / "work" / target.relative_to(KERNEL_HOME)
    return scratch_dir / "mounts" / target.relative_to("/")


def mount_env_name(kernel_path: str | os.PathLike[str]) -> str:
    """BACKENDAI_MOUNT_<NAME> for a mount, NAME being its last path component."""
    name = re.sub(r"[^A-Za-z0-9]", "_", PurePosixPath(kernel_path).name).upper()
    return f"BACKENDAI_MOUNT_{name}"


def rebase_env_value(value: str, mounts: Mapping[str, str]) -> str:
    """Rewrite a value that is a mount's kernel path, or a path under one, to the host path."""
    for kernel_path, host_path in mounts.items():
        if value == kernel_path or value.startswith(kernel_path + "/"):
            return host_path + value[len(kernel_path) :]
    return value


def rebase_kernel_path(command: str, kernel_path: str, host_path: str) -> str:
    """Replace kernel_path in a command where it is a whole path or the head of one."""
    # ponytail: the host path is not shell-quoted; a scratch root with spaces breaks the command.
    pattern = rf"(?<![\w./-]){re.escape(kernel_path.rstrip('/'))}(?=$|[/\s'\"])"
    return re.sub(pattern, lambda _: host_path, command)


def read_process_info(config_dir: Path) -> KernelProcessInfo | None:
    try:
        return KernelProcessInfo.model_validate_json(
            (config_dir / PROCESS_INFO_FILENAME).read_bytes()
        )
    except (OSError, ValueError):
        return None


def write_process_info(config_dir: Path, info: KernelProcessInfo) -> None:
    # Replaced atomically: a kernel with a truncated record could not be found again.
    tmp_path = config_dir / f"{PROCESS_INFO_FILENAME}.tmp"
    tmp_path.write_text(info.model_dump_json())
    tmp_path.replace(config_dir / PROCESS_INFO_FILENAME)


def _is_listening(port: int) -> bool:
    # ponytail: probes the loopback only; a listener bound to one other interface is missed.
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def find_port_conflicts(
    scratch_root: Path, kernel_id: KernelId, ports: Collection[int]
) -> list[int]:
    """Ports that another kernel of this host declares, or that something already listens on."""
    declared: set[int] = set()
    for info_path in scratch_root.glob(f"*/config/{PROCESS_INFO_FILENAME}"):
        if info_path.parents[1].name == str(kernel_id):
            continue
        # A record stays until its kernel is cleaned, so leftover children are covered too.
        if (info := read_process_info(info_path.parent)) is not None:
            declared.update(info.service_ports)
    return sorted(port for port in set(ports) if port in declared or _is_listening(port))


def _signal_group(pgid: int, sig: int) -> bool:
    """Return False when the signal reached no process in the group."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        # macOS reports EPERM when every member left is exiting or a zombie.
        return False
    return True


async def terminate_process_group(info: KernelProcessInfo, grace_period: float) -> None:
    """SIGTERM the kernel's process group, then SIGKILL what is left after the grace period."""
    if not info.owns_group() or not _signal_group(info.pid, signal.SIGTERM):
        return
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace_period
    while loop.time() < deadline:
        if not _signal_group(info.pid, 0) and not info.is_alive():
            return
        await asyncio.sleep(0.1)
    _signal_group(info.pid, signal.SIGKILL)
