from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from decimal import Decimal
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import psutil
import pytest

from ai.backend.agent.agent import ACTIVE_STATUS_SET, DEAD_STATUS_SET
from ai.backend.agent.alloc_map import DeviceSlotInfo, DiscretePropertyAllocMap
from ai.backend.agent.config.unified import AgentConfig
from ai.backend.agent.docker.intrinsic import CPUPlugin as DockerCPUPlugin
from ai.backend.agent.errors import InvalidMountPathError, PortConflictError
from ai.backend.agent.errors.backend import KernelRuntimeNotFoundError
from ai.backend.agent.kernel_registry.loader.pickle import PickleBasedKernelRegistryLoader
from ai.backend.agent.kernel_registry.writer.pickle import PickleBasedKernelRegistryWriter
from ai.backend.agent.kernel_registry.writer.types import KernelRegistrySaveMetadata
from ai.backend.agent.native import NativeAgentDiscovery
from ai.backend.agent.native.agent import (
    NativeAgent,
    NativeKernelCreationContext,
    link_mount,
    resolve_runtime_path,
)
from ai.backend.agent.native.intrinsic import MemoryPlugin
from ai.backend.agent.native.kernel import NativeKernel
from ai.backend.agent.native.process import (
    LOG_FILENAME,
    KernelProcessInfo,
    find_port_conflicts,
    host_path_of,
    mount_env_name,
    read_process_info,
    rebase_env_value,
    rebase_kernel_path,
    terminate_process_group,
    write_process_info,
)
from ai.backend.agent.resources import KernelResourceSpec, Mount
from ai.backend.agent.types import (
    AgentBackend,
    KernelLifecycleStatus,
    KernelOwnershipData,
    get_agent_discovery,
)
from ai.backend.common.docker import ImageRef, LabelName
from ai.backend.common.events.event_types.kernel.types import KernelLifecycleEventReason
from ai.backend.common.events.event_types.session.anycast import SessionFailureAnycastEvent
from ai.backend.common.types import (
    AgentId,
    ContainerStatus,
    DeviceId,
    DeviceName,
    KernelId,
    MountPermission,
    MountTypes,
    ResourceSlot,
    ServicePort,
    ServicePortProtocols,
    SessionId,
    SessionTypes,
    SlotName,
    SlotTypes,
)


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
    def scratch_dir(self, tmp_path: Path) -> Path:
        path = tmp_path / "scratch"
        (path / "work").mkdir(parents=True)
        return path

    @pytest.fixture
    def vfolder(self, tmp_path: Path) -> Path:
        path = tmp_path / "vfolder"
        path.mkdir()
        return path

    def test_links_home_path_into_work_dir(self, scratch_dir: Path, vfolder: Path) -> None:
        mount = Mount(MountTypes.BIND, vfolder, Path("/home/work/data/set1"))
        link_mount(scratch_dir, mount)
        link_mount(scratch_dir, mount)  # a kernel restart links again

        (scratch_dir / "work" / "data" / "set1" / "out.txt").write_text("x")
        assert (vfolder / "out.txt").read_text() == "x"

    @pytest.mark.parametrize(
        ("target", "link_path"),
        [
            ("/models", "mounts/models"),
            ("/data/set1", "mounts/data/set1"),
            ("/home/work/../config/x", "mounts/home/config/x"),
            ("/home/workspace/x", "mounts/home/workspace/x"),
        ],
    )
    def test_links_other_path_into_mounts_dir(
        self, scratch_dir: Path, vfolder: Path, target: str, link_path: str
    ) -> None:
        link_mount(scratch_dir, Mount(MountTypes.BIND, vfolder, Path(target)))

        assert (scratch_dir / link_path).resolve() == vfolder.resolve()
        assert host_path_of(scratch_dir, target) == scratch_dir / link_path
        assert list((scratch_dir / "work").iterdir()) == []

    @pytest.mark.parametrize("target", ["/home/work", "/home/work/", "/", "relative/path"])
    def test_rejects_unmappable_target(self, scratch_dir: Path, vfolder: Path, target: str) -> None:
        with pytest.raises(InvalidMountPathError):
            link_mount(scratch_dir, Mount(MountTypes.BIND, vfolder, Path(target)))
        assert [path.name for path in scratch_dir.iterdir()] == ["work"]

    def test_rejects_non_bind_mount(self, scratch_dir: Path) -> None:
        mount = Mount(
            MountTypes.VOLUME, Path("vol"), Path("/home/work/vol"), MountPermission.READ_WRITE
        )
        with pytest.raises(InvalidMountPathError):
            link_mount(scratch_dir, mount)


class TestEnvRewrite:
    MOUNTS = {"/home/work/data": "/s/work/data", "/models": "/s/mounts/models"}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("/models", "/s/mounts/models"),
            ("/models/sub/x.bin", "/s/mounts/models/sub/x.bin"),
            ("/home/work/data/train.jsonl", "/s/work/data/train.jsonl"),
            ("/models2", "/models2"),
            ("see /models here", "see /models here"),
            ("", ""),
        ],
    )
    def test_rewrites_only_values_that_are_mount_paths(self, value: str, expected: str) -> None:
        assert rebase_env_value(value, self.MOUNTS) == expected

    def test_mount_variable_is_named_by_the_last_component(self) -> None:
        assert mount_env_name("/home/work/tune-dataset") == "BACKENDAI_MOUNT_TUNE_DATASET"
        assert mount_env_name("/models/") == "BACKENDAI_MOUNT_MODELS"


class TestModelPathRebase:
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("vllm serve /models", "vllm serve /host/m"),
            ("serve --model '/models' --port 8080", "serve --model '/host/m' --port 8080"),
            ("serve --model=/models/sub --x", "serve --model=/host/m/sub --x"),
            ("serve --probe /v1/models --dir /models2", "serve --probe /v1/models --dir /models2"),
        ],
    )
    def test_replaces_only_whole_path_components(self, command: str, expected: str) -> None:
        assert rebase_kernel_path(command, "/models", "/host/m") == expected
        assert rebase_kernel_path(command, "/models/", "/host/m") == expected

    async def test_model_service_is_fed_host_paths(self, tmp_path: Path) -> None:
        kernel = MagicMock(spec=NativeKernel)
        kernel._scratch_dir = tmp_path
        runner = kernel._runner.return_value
        runner.feed_start_model_service = AsyncMock(return_value={"status": "started"})
        model_service = {
            "name": "m",
            "model_path": "/models",
            "service": {"start_command": "serve --model /models", "port": 8080},
        }

        await NativeKernel.start_model_service(kernel, model_service)

        host_path = str(tmp_path / "mounts" / "models")
        runner.feed_start_model_service.assert_awaited_once_with({
            "name": "m",
            "model_path": host_path,
            "service": {"start_command": f"serve --model {host_path}", "port": 8080},
        })
        assert model_service["model_path"] == "/models"  # the caller's definition is untouched


class TestServicePorts:
    @pytest.fixture
    def listening_port(self) -> Iterator[int]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            yield sock.getsockname()[1]

    @pytest.fixture
    def free_port(self) -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port: int = sock.getsockname()[1]
        return port

    def _record(self, scratch_root: Path, kernel_id: KernelId, service_ports: list[int]) -> None:
        config_dir = scratch_root / str(kernel_id) / "config"
        config_dir.mkdir(parents=True)
        info = KernelProcessInfo(
            container_id="c" * 64,
            pid=os.getpid(),
            create_time=0.0,
            image="local/stable/python:3.12-macos",
            labels={},
            host_ports=[],
            service_ports=service_ports,
        )
        write_process_info(config_dir, info)

    def test_port_with_a_listener_conflicts(
        self, tmp_path: Path, listening_port: int, free_port: int
    ) -> None:
        conflicts = find_port_conflicts(tmp_path, KernelId(uuid4()), [listening_port, free_port])
        assert conflicts == [listening_port]

    def test_port_declared_by_another_kernel_conflicts_until_cleaned(
        self, tmp_path: Path, free_port: int
    ) -> None:
        owner, other = KernelId(uuid4()), KernelId(uuid4())
        self._record(tmp_path, owner, [free_port])

        assert find_port_conflicts(tmp_path, other, [free_port]) == [free_port]
        assert find_port_conflicts(tmp_path, owner, [free_port]) == []  # e.g., its own restart

        (tmp_path / str(owner) / "config" / "native-kernel.json").unlink()
        assert find_port_conflicts(tmp_path, other, [free_port]) == []

    def _sport(self, name: str, port: int) -> ServicePort:
        return {
            "name": name,
            "protocol": ServicePortProtocols.PREOPEN,
            "container_ports": (port,),
            "host_ports": (None,),
            "is_inference": False,
        }

    async def test_conflict_fails_creation_before_anything_is_acquired(
        self, tmp_path: Path, listening_port: int
    ) -> None:
        ctx = MagicMock(spec=NativeKernelCreationContext)
        ctx.kernel_id = KernelId(uuid4())
        ctx.local_config = MagicMock()
        ctx.local_config.container.scratch_root = tmp_path
        ctx.service_port_lock = asyncio.Lock()
        ctx.port_pool = MagicMock()
        kernel = MagicMock(service_ports=[self._sport("web", listening_port)])

        with pytest.raises(PortConflictError):
            await NativeKernelCreationContext.start_container(
                ctx,
                kernel,
                ["/opt/backend.ai/bin/python", "-m", "ai.backend.kernel"],
                None,
                [],
                MagicMock(),
            )

        ctx.port_pool.acquire.assert_not_called()
        ctx._spawn.assert_not_called()

    async def test_container_only_services_are_not_advertised(self) -> None:
        ctx = MagicMock(restarting=True)
        service_ports = [
            self._sport("sshd", 2200),
            self._sport("ttyd", 7681),
            self._sport("web", 8080),
        ]

        with patch("ai.backend.agent.native.agent.NativeKernel") as kernel_cls:
            await NativeKernelCreationContext.prepare_container(
                ctx, MagicMock(), {}, service_ports, MagicMock()
            )

        assert [sport["name"] for sport in service_ports] == ["web"]
        assert kernel_cls.call_args.kwargs["service_ports"] == service_ports


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
        info = NativeKernelCreationContext._spawn(
            ctx, [sys.executable, "-c", code], [30000, 30001], [8080]
        )
        try:
            assert read_process_info(ctx.config_dir) == info
            assert info.service_ports == [8080]
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


class TestRestartRecovery:
    AGENT_ID = AgentId("i-test")

    @pytest.fixture
    def scratch_root(self, tmp_path: Path) -> Path:
        path = tmp_path / "scratches"
        path.mkdir()
        return path

    @pytest.fixture
    def agent(self, tmp_path: Path, scratch_root: Path) -> MagicMock:
        registry_path = tmp_path / "last_registry.dat"
        agent = MagicMock(spec=NativeAgent)
        agent.id = self.AGENT_ID
        agent._scratch_root = scratch_root
        agent._registry_loader = PickleBasedKernelRegistryLoader(
            registry_path, tmp_path / "legacy.dat"
        )
        agent._registry_writer = PickleBasedKernelRegistryWriter(registry_path)
        agent._read_process_info = partial(NativeAgent._read_process_info, agent)
        agent._mark_exit_handled = partial(NativeAgent._mark_exit_handled, agent)
        agent.anycast_and_broadcast_event = AsyncMock()
        return agent

    @pytest.fixture
    def live_process(self) -> Iterator[psutil.Process]:
        proc = subprocess.Popen(["/bin/sh", "-c", "sleep 600"], start_new_session=True)
        threading.Thread(target=proc.wait, daemon=True).start()
        yield psutil.Process(proc.pid)
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass

    def _record(
        self,
        scratch_root: Path,
        proc: psutil.Process,
        *,
        create_time: float | None = None,
        owner_agent: str = AGENT_ID,
        exit_handled: bool = False,
    ) -> tuple[KernelId, SessionId, Path]:
        kernel_id, session_id = KernelId(uuid4()), SessionId(uuid4())
        config_dir = scratch_root / str(kernel_id) / "config"
        config_dir.mkdir(parents=True)
        info = KernelProcessInfo(
            container_id="c" * 64,
            pid=proc.pid,
            create_time=proc.create_time() if create_time is None else create_time,
            image="local/stable/python:3.12-macos",
            labels={
                LabelName.KERNEL_ID.value: str(kernel_id),
                LabelName.SESSION_ID.value: str(session_id),
                LabelName.OWNER_AGENT.value: owner_agent,
            },
            host_ports=[30000, 30001],
            service_ports=[8080],
            exit_handled=exit_handled,
        )
        write_process_info(config_dir, info)
        return kernel_id, session_id, config_dir

    def _kernel(self, scratch_root: Path) -> NativeKernel:
        kernel_id = KernelId(uuid4())
        kernel = NativeKernel(
            KernelOwnershipData(kernel_id, SessionId(uuid4()), self.AGENT_ID),
            "",
            ImageRef(
                name="python",
                project="stable",
                tag="3.12-macos",
                registry="local",
                architecture="aarch64",
                is_local=True,
            ),
            1,
            agent_config={"container": {"scratch-root": scratch_root}},
            resource_spec=KernelResourceSpec(
                slots=ResourceSlot({"cpu": Decimal(1)}), allocations={}, scratch_disk_size=0
            ),
            service_ports=[],
            data={"repl_in_port": 30000, "repl_out_port": 30001, "host_ports": [30000, 30001]},
            environ={},
            session_type=SessionTypes.BATCH,
        )
        kernel.state = KernelLifecycleStatus.RUNNING
        return kernel

    async def test_registry_survives_a_restart(self, agent: MagicMock, scratch_root: Path) -> None:
        assert await NativeAgent._load_kernel_registry_from_recovery(agent) == {}  # first start

        kernel = self._kernel(scratch_root)
        # Not forced by the caller, and well within the writer's cool-down.
        await NativeAgent._write_kernel_registry_to_recovery(
            agent, {kernel.kernel_id: kernel}, KernelRegistrySaveMetadata(force=False)
        )

        restored = await NativeAgent._load_kernel_registry_from_recovery(agent)
        restored_kernel = restored[kernel.kernel_id]
        assert isinstance(restored_kernel, NativeKernel)
        assert restored_kernel.session_id == kernel.session_id
        assert restored_kernel.session_type == SessionTypes.BATCH
        assert restored_kernel.state == KernelLifecycleStatus.RUNNING
        assert restored_kernel["host_ports"] == [30000, 30001]

    async def test_unreadable_registry_starts_empty(self, agent: MagicMock, tmp_path: Path) -> None:
        (tmp_path / "last_registry.dat").write_bytes(b"")
        assert await NativeAgent._load_kernel_registry_from_recovery(agent) == {}

    async def test_records_map_to_containers_by_liveness(
        self, agent: MagicMock, scratch_root: Path, live_process: psutil.Process
    ) -> None:
        live, _, _ = self._record(scratch_root, live_process)
        reused_pid, _, _ = self._record(
            scratch_root, live_process, create_time=live_process.create_time() - 3600
        )
        gone = subprocess.Popen(["/usr/bin/true"])
        gone.wait()
        gone_proc = MagicMock(pid=gone.pid, create_time=lambda: 0.0)
        dead, _, _ = self._record(scratch_root, gone_proc)
        self._record(scratch_root, live_process, owner_agent="i-other")

        active = await NativeAgent.enumerate_containers(agent, ACTIVE_STATUS_SET)
        every = await NativeAgent.enumerate_containers(agent, ACTIVE_STATUS_SET | DEAD_STATUS_SET)

        assert [kernel_id for kernel_id, _ in active] == [live]
        statuses = {kernel_id: container.status for kernel_id, container in every}
        assert statuses == {
            live: ContainerStatus.RUNNING,
            reused_pid: ContainerStatus.EXITED,
            dead: ContainerStatus.EXITED,
        }
        # The agent restores its port pool from these.
        assert [port.host_port for port in dict(active)[live].ports] == [30000, 30001]

    async def test_allocations_are_rebuilt_from_the_record(
        self, scratch_root: Path, live_process: psutil.Process
    ) -> None:
        _, _, config_dir = self._record(scratch_root, live_process)
        cpu, mem = DeviceName("cpu"), DeviceName("mem")
        spec = KernelResourceSpec(
            slots=ResourceSlot({"cpu": Decimal(2), "mem": Decimal(1024)}),
            allocations={
                cpu: {SlotName("cpu"): {DeviceId("0"): Decimal(1), DeviceId("1"): Decimal(1)}},
                mem: {SlotName("mem"): {DeviceId("root"): Decimal(1024)}},
            },
            scratch_disk_size=0,
        )
        with (config_dir / "resource.txt").open("w") as f:
            spec.write_to_file(f)
        info = read_process_info(config_dir)
        assert info is not None
        container = info.to_container(config_dir)
        cpu_map = DiscretePropertyAllocMap(
            device_slots={
                DeviceId(str(idx)): DeviceSlotInfo(SlotTypes.COUNT, SlotName("cpu"), Decimal(1))
                for idx in range(4)
            }
        )
        mem_map = DiscretePropertyAllocMap(
            device_slots={
                DeviceId("root"): DeviceSlotInfo(SlotTypes.BYTES, SlotName("mem"), Decimal(4096))
            }
        )

        await DockerCPUPlugin.restore_from_container(MagicMock(), container, cpu_map)
        await MemoryPlugin.restore_from_container(MagicMock(), container, mem_map)

        assert cpu_map.allocations[SlotName("cpu")] == {
            DeviceId("0"): Decimal(1),
            DeviceId("1"): Decimal(1),
            DeviceId("2"): Decimal(0),
            DeviceId("3"): Decimal(0),
        }
        assert mem_map.allocations[SlotName("mem")] == {DeviceId("root"): Decimal(1024)}


class TestLiveness:
    """A clean that no destroy preceded means the runner died on its own."""

    @pytest.fixture
    def agent(self, tmp_path: Path) -> MagicMock:
        agent = MagicMock(spec=NativeAgent)
        agent._scratch_root = tmp_path
        agent._read_process_info = partial(NativeAgent._read_process_info, agent)
        agent._mark_exit_handled = partial(NativeAgent._mark_exit_handled, agent)
        agent.anycast_and_broadcast_event = AsyncMock()
        agent.collect_logs = AsyncMock()
        return agent

    @pytest.fixture
    def dead_kernel(self, tmp_path: Path) -> tuple[KernelId, SessionId]:
        kernel_id, session_id = KernelId(uuid4()), SessionId(uuid4())
        config_dir = tmp_path / str(kernel_id) / "config"
        config_dir.mkdir(parents=True)
        gone = subprocess.Popen(["/usr/bin/true"], start_new_session=True)
        gone.wait()
        info = KernelProcessInfo(
            container_id="c" * 64,
            pid=gone.pid,
            create_time=0.0,
            image="local/stable/python:3.12-macos",
            labels={LabelName.SESSION_ID.value: str(session_id)},
            host_ports=[30000, 30001],
            service_ports=[8080],
        )
        write_process_info(config_dir, info)
        return kernel_id, session_id

    async def test_death_is_reported_as_a_session_failure(
        self, agent: MagicMock, dead_kernel: tuple[KernelId, SessionId], tmp_path: Path
    ) -> None:
        kernel_id, session_id = dead_kernel

        await NativeAgent.clean_kernel(agent, kernel_id, None, False)

        anycast_event, _ = agent.anycast_and_broadcast_event.await_args.args
        assert isinstance(anycast_event, SessionFailureAnycastEvent)
        assert anycast_event.session_id == UUID(str(session_id))
        assert anycast_event.reason == KernelLifecycleEventReason.SELF_TERMINATED
        # The record goes with the scratch dir, which ends its claim on the service ports.
        assert not (tmp_path / str(kernel_id)).exists()

    async def test_destroyed_kernel_is_not_reported(
        self, agent: MagicMock, dead_kernel: tuple[KernelId, SessionId], tmp_path: Path
    ) -> None:
        kernel_id, _ = dead_kernel

        await NativeAgent.destroy_kernel(agent, kernel_id, None)
        info = read_process_info(tmp_path / str(kernel_id) / "config")
        assert info is not None and info.exit_handled  # survives an agent restart in between
        await NativeAgent.clean_kernel(agent, kernel_id, None, False)

        agent.anycast_and_broadcast_event.assert_not_awaited()
        assert not (tmp_path / str(kernel_id)).exists()

    async def test_restarting_kernel_is_not_reported(
        self, agent: MagicMock, dead_kernel: tuple[KernelId, SessionId], tmp_path: Path
    ) -> None:
        kernel_id, _ = dead_kernel

        await NativeAgent.clean_kernel(agent, kernel_id, None, True)

        agent.anycast_and_broadcast_event.assert_not_awaited()
        assert (tmp_path / str(kernel_id)).exists()

    async def test_death_is_reported_once(
        self, agent: MagicMock, dead_kernel: tuple[KernelId, SessionId]
    ) -> None:
        kernel_id, _ = dead_kernel
        with patch("ai.backend.agent.native.agent.shutil.rmtree"):  # a clean still in flight
            await NativeAgent.clean_kernel(agent, kernel_id, None, False)
            await NativeAgent.clean_kernel(agent, kernel_id, None, False)

        agent.anycast_and_broadcast_event.assert_awaited_once()
