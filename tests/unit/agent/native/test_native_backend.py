from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import psutil
import pytest

from ai.backend.agent.config.unified import AgentConfig
from ai.backend.agent.errors import InvalidMountPathError
from ai.backend.agent.errors.backend import KernelRuntimeNotFoundError
from ai.backend.agent.native import NativeAgentDiscovery
from ai.backend.agent.native.agent import (
    NativeAgent,
    NativeKernelCreationContext,
    link_mount,
    resolve_runtime_path,
)
from ai.backend.agent.native.process import (
    LOG_FILENAME,
    KernelProcessInfo,
    read_process_info,
    terminate_process_group,
    write_process_info,
)
from ai.backend.agent.resources import Mount
from ai.backend.agent.types import AgentBackend, get_agent_discovery
from ai.backend.common.docker import LabelName
from ai.backend.common.types import ContainerStatus, MountPermission, MountTypes


class TestBackendSelection:
    def test_config_accepts_native_backend(self) -> None:
        config = AgentConfig.model_validate({
            "backend": "native",
            "rpc-listen-addr": {"host": "127.0.0.1", "port": 6001},
        })
        assert config.backend == AgentBackend.NATIVE

    def test_discovery_resolves_native_agent(self) -> None:
        discovery = get_agent_discovery(AgentBackend.NATIVE)
        assert isinstance(discovery, NativeAgentDiscovery)
        assert discovery.get_agent_cls() is NativeAgent


class TestResolveRuntimePath:
    def test_returns_host_executable(self) -> None:
        labels = {LabelName.RUNTIME_PATH.value: sys.executable}
        assert resolve_runtime_path(labels) == Path(sys.executable)

    @pytest.mark.parametrize(
        "labels",
        [
            {},
            {LabelName.RUNTIME_PATH.value: "bin/python"},
            {LabelName.RUNTIME_PATH.value: "/nonexistent/bin/python"},
        ],
        ids=["missing label", "relative path", "not on this host"],
    )
    def test_rejects_unusable_runtime(self, labels: dict[str, str]) -> None:
        with pytest.raises(KernelRuntimeNotFoundError):
            resolve_runtime_path(labels)

    def test_rejects_non_executable_file(self, tmp_path: Path) -> None:
        plain_file = tmp_path / "python"
        plain_file.write_text("")
        with pytest.raises(KernelRuntimeNotFoundError):
            resolve_runtime_path({LabelName.RUNTIME_PATH.value: str(plain_file)})


class TestLinkMount:
    @pytest.fixture
    def work_dir(self, tmp_path: Path) -> Path:
        path = tmp_path / "work"
        path.mkdir()
        return path

    @pytest.fixture
    def vfolder(self, tmp_path: Path) -> Path:
        path = tmp_path / "vfolder"
        path.mkdir()
        return path

    def test_links_at_path_relative_to_kernel_home(self, work_dir: Path, vfolder: Path) -> None:
        mount = Mount(MountTypes.BIND, vfolder, Path("/home/work/data/set1"))
        link_mount(work_dir, mount)
        link_mount(work_dir, mount)  # a kernel restart links again

        (work_dir / "data" / "set1" / "out.txt").write_text("x")
        assert (vfolder / "out.txt").read_text() == "x"

    @pytest.mark.parametrize(
        "target",
        ["/home/work", "/data", "/home/work/../config/x", "/home/workspace/x"],
    )
    def test_rejects_target_outside_kernel_home(
        self, work_dir: Path, vfolder: Path, target: str
    ) -> None:
        with pytest.raises(InvalidMountPathError):
            link_mount(work_dir, Mount(MountTypes.BIND, vfolder, Path(target)))
        assert list(work_dir.iterdir()) == []

    def test_rejects_non_bind_mount(self, work_dir: Path) -> None:
        mount = Mount(
            MountTypes.VOLUME, Path("vol"), Path("/home/work/vol"), MountPermission.READ_WRITE
        )
        with pytest.raises(InvalidMountPathError):
            link_mount(work_dir, mount)


class TestKernelProcess:
    @pytest.fixture
    def kernel_process(self) -> Iterator[KernelProcessInfo]:
        # A runner stand-in: a session leader with a child that outlives SIGTERM to the leader.
        proc = subprocess.Popen(["/bin/sh", "-c", "sleep 600 & wait"], start_new_session=True)
        threading.Thread(target=proc.wait, daemon=True).start()
        deadline = time.monotonic() + 5.0
        while not psutil.Process(proc.pid).children() and time.monotonic() < deadline:
            time.sleep(0.01)
        yield KernelProcessInfo(
            container_id="c" * 64,
            pid=proc.pid,
            create_time=psutil.Process(proc.pid).create_time(),
            image="local/stable/python:3.12-macos",
            labels={LabelName.KERNEL_ID.value: "00000000-0000-0000-0000-000000000001"},
            host_ports=[30000, 30001],
        )
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass

    def test_record_round_trips_through_config_dir(
        self, kernel_process: KernelProcessInfo, tmp_path: Path
    ) -> None:
        assert read_process_info(tmp_path) is None
        write_process_info(tmp_path, kernel_process)
        assert read_process_info(tmp_path) == kernel_process

    async def test_terminate_kills_the_whole_group(
        self, kernel_process: KernelProcessInfo, tmp_path: Path
    ) -> None:
        child = psutil.Process(kernel_process.pid).children()[0]
        assert kernel_process.to_container(tmp_path).status == ContainerStatus.RUNNING

        await terminate_process_group(kernel_process, grace_period=5.0)

        assert not kernel_process.is_alive()
        _, still_running = psutil.wait_procs([child], timeout=5.0)
        assert not still_running
        container = kernel_process.to_container(tmp_path)
        assert container.status == ContainerStatus.EXITED
        assert [port.host_port for port in container.ports] == [30000, 30001]

    async def test_reused_pid_is_neither_alive_nor_signalled(
        self, kernel_process: KernelProcessInfo
    ) -> None:
        stale = kernel_process.model_copy(update={"create_time": kernel_process.create_time - 3600})
        assert not stale.is_alive()
        assert not stale.owns_group()

        await terminate_process_group(stale, grace_period=0.1)

        assert kernel_process.is_alive()


class TestSpawn:
    @pytest.fixture
    def ctx(self, tmp_path: Path) -> MagicMock:
        ctx = MagicMock(spec=NativeKernelCreationContext)
        ctx.config_dir = tmp_path / "config"
        ctx.work_dir = tmp_path / "work"
        ctx.config_dir.mkdir()
        ctx.work_dir.mkdir()
        ctx.agent_id = "i-test"
        ctx.kernel_id = uuid4()
        ctx.session_id = uuid4()
        ctx.kspec_version = 1
        ctx.ownership_data = MagicMock(owner_user_id_to_str=None, owner_project_id_to_str=None)
        ctx.image_ref = MagicMock(canonical="local/stable/python:3.12-macos")
        ctx._runner_env.return_value = {"PATH": os.defpath, "MARKER": "from-agent"}
        return ctx

    async def test_spawn_records_a_detached_process(self, ctx: MagicMock) -> None:
        code = (
            "import os, time; print(os.getcwd(), os.environ['MARKER'], flush=True); time.sleep(600)"
        )
        info = NativeKernelCreationContext._spawn(ctx, [sys.executable, "-c", code], [30000, 30001])
        try:
            assert read_process_info(ctx.config_dir) == info
            assert os.getpgid(info.pid) == info.pid != os.getpgid(0)
            assert json.loads((ctx.config_dir / "intrinsic-ports.json").read_text()) == {
                "replin": 30000,
                "replout": 30001,
            }
            container = info.to_container(ctx.config_dir)
            assert container.kernel_id == ctx.kernel_id
            assert container.session_id == ctx.session_id
            log_path = ctx.config_dir / LOG_FILENAME
            deadline = time.monotonic() + 5.0
            while not log_path.read_text() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert log_path.read_text().split() == [str(ctx.work_dir.resolve()), "from-agent"]
        finally:
            await terminate_process_group(info, grace_period=5.0)
        assert not info.is_alive()


class TestWriteDotfiles:
    def test_writes_only_inside_the_work_dir(self, tmp_path: Path) -> None:
        ctx = MagicMock(spec=NativeKernelCreationContext)
        ctx.work_dir = tmp_path / "work"
        ctx.work_dir.mkdir()
        outside = tmp_path / "outside.txt"
        ctx.internal_data = {
            "dotfiles": [
                {"path": ".config/app.toml", "data": "a = 1", "perm": "644"},
                {"path": "/home/work/.netrc", "data": "machine x\n", "perm": "600"},
                {"path": str(outside), "data": "x", "perm": "644"},
                {"path": "../outside.txt", "data": "x", "perm": "644"},
            ]
        }

        NativeKernelCreationContext._write_dotfiles(ctx)

        assert (ctx.work_dir / ".config" / "app.toml").read_text() == "a = 1\n"
        netrc = ctx.work_dir / ".netrc"
        assert netrc.read_text() == "machine x\n"
        assert netrc.stat().st_mode & 0o777 == 0o600
        assert not outside.exists()
