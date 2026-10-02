"""On-disk record and signalling of a kernel that runs as a host process tree."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Final

import psutil
from pydantic import BaseModel

from ai.backend.agent.types import Container, Port
from ai.backend.common.types import ContainerId, ContainerStatus, KernelId

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


def read_process_info(config_dir: Path) -> KernelProcessInfo | None:
    try:
        return KernelProcessInfo.model_validate_json(
            (config_dir / PROCESS_INFO_FILENAME).read_bytes()
        )
    except (OSError, ValueError):
        return None


def write_process_info(config_dir: Path, info: KernelProcessInfo) -> None:
    (config_dir / PROCESS_INFO_FILENAME).write_text(info.model_dump_json())


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
