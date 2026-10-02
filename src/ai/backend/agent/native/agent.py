from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import AsyncGenerator, Mapping, MutableMapping, Sequence
from contextlib import suppress
from decimal import Decimal
from importlib.resources import files
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any, Final, override
from uuid import UUID

import psutil

from ai.backend.agent.agent import (
    ACTIVE_STATUS_SET,
    AbstractAgent,
    AbstractKernelCreationContext,
    ScanImagesResult,
)
from ai.backend.agent.config.unified import AgentUnifiedConfig
from ai.backend.agent.errors import (
    InvalidArgumentError,
    InvalidMountPathError,
    PortConflictError,
    UnsupportedResource,
)
from ai.backend.agent.errors.backend import (
    KernelRuntimeNotFoundError,
    UnsupportedBackendOperationError,
)
from ai.backend.agent.kernel import AbstractKernel
from ai.backend.agent.kernel_registry.exception import (
    KernelRegistryLoadError,
    KernelRegistryNotFound,
)
from ai.backend.agent.kernel_registry.loader.pickle import PickleBasedKernelRegistryLoader
from ai.backend.agent.kernel_registry.writer.pickle import PickleBasedKernelRegistryWriter
from ai.backend.agent.kernel_registry.writer.types import KernelRegistrySaveMetadata
from ai.backend.agent.port_pool import PortPool
from ai.backend.agent.resources import (
    AbstractComputePlugin,
    ComputerContext,
    KernelResourceSpec,
    Mount,
    known_slot_types,
)
from ai.backend.agent.types import Container, KernelOwnershipData, MountInfo
from ai.backend.common.asyncio import run_in_executor_with_context
from ai.backend.common.cgroup import CgroupController
from ai.backend.common.docker import ImageRef, LabelName
from ai.backend.common.dto.agent.response import PurgeImagesResp
from ai.backend.common.dto.manager.rpc_request import PurgeImagesReq
from ai.backend.common.events.dispatcher import EventProducer
from ai.backend.common.events.event_types.kernel.types import KernelLifecycleEventReason
from ai.backend.common.events.event_types.session.anycast import SessionFailureAnycastEvent
from ai.backend.common.events.event_types.session.broadcast import SessionFailureBroadcastEvent
from ai.backend.common.json import dump_json
from ai.backend.common.types import (
    AutoPullBehavior,
    ClusterInfo,
    ClusterSSHPortMapping,
    ContainerId,
    ContainerStatus,
    DeviceId,
    DeviceName,
    ImageConfig,
    ImageRegistry,
    KernelCreationConfig,
    KernelId,
    MountPermission,
    MountTypes,
    ResourceSlot,
    Sentinel,
    ServicePort,
    SessionId,
    SlotName,
    current_resource_slots,
)
from ai.backend.logging.structured import StructuredLogger

from .kernel import NativeKernel
from .process import (
    KERNEL_HOME,
    LOG_FILENAME,
    PROCESS_INFO_FILENAME,
    KernelProcessInfo,
    config_dir_of,
    find_port_conflicts,
    host_path_of,
    read_process_info,
    terminate_process_group,
    write_process_info,
)

log = StructuredLogger(logging.getLogger(__spec__.name))

# create_kernel() addresses the runner by its in-container interpreter path.
_CONTAINER_KRUNNER_PYTHON: Final = "/opt/backend.ai/bin/python"
_DEFAULT_PATH: Final = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
_PASSTHROUGH_ENVS: Final = ("LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME")
_TERMINATION_GRACE_PERIOD: Final = 10.0  # seconds, as `docker stop`
# create_kernel() lists them for every kernel, but they need the container-side runner binaries.
_CONTAINER_ONLY_SERVICES: Final = frozenset({"sshd", "ttyd"})


def resolve_runtime_path(image_labels: Mapping[str, str]) -> Path:
    """The image is metadata only: its runtime path must be an executable on this host."""
    raw_path = image_labels.get(LabelName.RUNTIME_PATH)
    if not raw_path:
        raise KernelRuntimeNotFoundError(
            f"The image must set the {LabelName.RUNTIME_PATH.value} label to a host path."
        )
    runtime_path = Path(raw_path)
    if not (runtime_path.is_absolute() and os.access(runtime_path, os.X_OK)):
        raise KernelRuntimeNotFoundError(f"{runtime_path} is not an executable on this host.")
    return runtime_path


def link_mount(scratch_dir: Path, mount: Mount) -> None:
    """Expose a mount as a symlink in the scratch dir, at host_path_of() its kernel path."""
    target = PurePosixPath(os.path.normpath(mount.target))
    if (
        mount.type != MountTypes.BIND
        or mount.source is None
        or not target.is_absolute()
        or target in (KERNEL_HOME, PurePosixPath("/"))
    ):
        raise InvalidMountPathError(
            "A host-process kernel can only mount host paths at an absolute path "
            f"other than / and {KERNEL_HOME} (got {mount.source} -> {mount.target})"
        )
    link_path = host_path_of(scratch_dir, target)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():  # re-created on kernel restarts
        link_path.unlink()
    link_path.symlink_to(mount.source)


class NativeKernelCreationContext(AbstractKernelCreationContext[NativeKernel]):
    scratch_dir: Path
    config_dir: Path
    work_dir: Path
    runtime_path: Path
    port_pool: PortPool
    service_port_lock: asyncio.Lock

    def __init__(
        self,
        ownership_data: KernelOwnershipData,
        event_producer: EventProducer,
        kernel_image: ImageRef,
        kernel_config: KernelCreationConfig,
        distro: str,
        local_config: AgentUnifiedConfig,
        computers: Mapping[DeviceName, ComputerContext],
        port_pool: PortPool,
        service_port_lock: asyncio.Lock,
        restarting: bool = False,
    ) -> None:
        super().__init__(
            ownership_data,
            event_producer,
            kernel_image,
            kernel_config,
            distro,
            local_config,
            computers,
            restarting=restarting,
        )
        self.config_dir = config_dir_of(local_config.container.scratch_root, self.kernel_id)
        self.scratch_dir = self.config_dir.parent
        self.work_dir = self.scratch_dir / "work"
        self.runtime_path = resolve_runtime_path(self.image_labels)
        self.port_pool = port_pool
        self.service_port_lock = service_port_lock

    @override
    async def get_extra_envs(self) -> Mapping[str, str]:
        return {}

    @override
    async def prepare_resource_spec(self) -> tuple[KernelResourceSpec, Mapping[str, Any] | None]:
        if self.restarting:

            def _read() -> KernelResourceSpec:
                with (self.config_dir / "resource.txt").open() as f:
                    return KernelResourceSpec.read_from_file(f)

            return await run_in_executor_with_context(None, _read), None
        slots = ResourceSlot.from_json(self.kernel_config["resource_slots"])
        for required in (SlotName("cpu"), SlotName("mem")):
            if required not in slots:
                raise UnsupportedResource(f"{required} slot is required")
        for slot_name, slot_value in slots.items():
            if slot_name not in known_slot_types and slot_value != Decimal(0):
                raise UnsupportedResource(slot_name)
        current_resource_slots.set(known_slot_types)
        slots = slots.normalize_slots(ignore_unknown=True)
        resource_spec = KernelResourceSpec(
            allocations={},
            slots=slots.copy(),
            mounts=[],
            scratch_disk_size=0,
        )
        return resource_spec, self.kernel_config.get("resource_opts", {})

    @override
    async def prepare_scratch(self) -> None:
        def _create_scratch_dirs() -> None:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.work_dir.mkdir(parents=True, exist_ok=True)

        await run_in_executor_with_context(None, _create_scratch_dirs)

    @override
    async def get_intrinsic_mounts(self) -> Sequence[Mount]:
        return []

    @property
    @override
    def repl_ports(self) -> Sequence[int]:
        return (2000, 2001)

    @property
    @override
    def protected_services(self) -> Sequence[str]:
        return ()

    @override
    async def apply_network(self, cluster_info: ClusterInfo) -> None:
        pass

    @override
    async def prepare_ssh(self, cluster_info: ClusterInfo) -> None:
        pass

    @override
    async def process_mounts(self, mounts: Sequence[Mount]) -> None:
        def _link_all() -> None:
            for mount in mounts:
                link_mount(self.scratch_dir, mount)

        await run_in_executor_with_context(None, _link_all)

    @override
    async def apply_accelerator_allocation(
        self,
        computer: AbstractComputePlugin,
        device_alloc: Mapping[SlotName, Mapping[DeviceId, Decimal]],
    ) -> None:
        pass

    @override
    async def generate_accelerator_mounts(
        self,
        computer: AbstractComputePlugin,
        device_alloc: Mapping[SlotName, Mapping[DeviceId, Decimal]],
    ) -> list[MountInfo]:
        return []

    @override
    def resolve_krunner_filepath(self, filename: str) -> Path:
        return Path(str(files("ai.backend.runner").joinpath("../" + filename))).resolve()

    @override
    def get_runner_mount(
        self,
        type: MountTypes,
        src: str | Path,
        target: str | Path,
        perm: MountPermission = MountPermission.READ_ONLY,
        opts: Mapping[str, Any] | None = None,
    ) -> Mount:
        return Mount(type, Path(src), Path(target), MountPermission(perm), opts=opts)

    @override
    async def mount_krunner(
        self,
        resource_spec: KernelResourceSpec,
        environ: MutableMapping[str, str],
    ) -> None:
        # The runner is the agent's own package; no binaries or hook libraries to inject.
        pass

    def _write_dotfiles(self) -> None:
        for dotfile in self.internal_data.get("dotfiles", []):
            path = PurePosixPath(dotfile["path"])
            if path.is_absolute():
                if not path.is_relative_to(KERNEL_HOME):
                    log.trace("dotfile outside the kernel home skipped", file_path=str(path))
                    continue
                path = path.relative_to(KERNEL_HOME)
            file_path = Path(os.path.normpath(self.work_dir / path))
            if not file_path.is_relative_to(self.work_dir):
                log.trace("dotfile outside the kernel home skipped", file_path=str(path))
                continue
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content: str = dotfile["data"]
            file_path.write_text(content if content.endswith("\n") else content + "\n")
            file_path.chmod(int(dotfile["perm"], 8))

    @override
    async def prepare_container(
        self,
        resource_spec: KernelResourceSpec,
        environ: Mapping[str, str],
        service_ports: list[ServicePort],
        cluster_info: ClusterInfo,
    ) -> NativeKernel:
        # In place: create_kernel() reports this very list to the manager.
        service_ports[:] = [
            sport for sport in service_ports if sport["name"] not in _CONTAINER_ONLY_SERVICES
        ]
        if not self.restarting:
            kernel_environ = {
                # Keep the runner's own import path out of user processes.
                "PYTHONPATH": "",
                **environ,
                "PATH": f"{self.runtime_path.parent}:{environ.get('PATH', _DEFAULT_PATH)}",
            }
            if model_path := environ.get("BACKEND_MODEL_PATH"):
                kernel_environ["BACKEND_MODEL_PATH"] = str(
                    host_path_of(self.scratch_dir, model_path)
                )
            with StringIO() as buf:
                resource_spec.write_to_file(buf)
                for dev_type, device_alloc in resource_spec.allocations.items():
                    device_plugin = self.computers[dev_type].instance
                    kvpairs = await device_plugin.generate_resource_data(device_alloc)
                    for k, v in kvpairs.items():
                        buf.write(f"{k}={v}\n")
                resource_txt = buf.getvalue()

            def _write_config() -> None:
                if bootstrap := self.kernel_config.get("bootstrap_script"):
                    (self.work_dir / "bootstrap.sh").write_text(bootstrap)
                (self.config_dir / "environ.txt").write_text(
                    "".join(f"{k}={v}\n" for k, v in kernel_environ.items())
                )
                (self.config_dir / "resource.txt").write_text(resource_txt)
                self._write_dotfiles()

            await run_in_executor_with_context(None, _write_config)

        return NativeKernel(
            self.ownership_data,
            self.kernel_config["network_id"],
            self.image_ref,
            self.kspec_version,
            agent_config=self.local_config.model_dump(by_alias=True),
            service_ports=service_ports,
            resource_spec=resource_spec,
            environ=environ,
            data={},
        )

    def _runner_env(self) -> dict[str, str]:
        runner_root = self.resolve_krunner_filepath("kernel").parents[2]
        return {
            **{k: v for k in _PASSTHROUGH_ENVS if (v := os.environ.get(k)) is not None},
            "PATH": f"{self.runtime_path.parent}:{_DEFAULT_PATH}",
            "HOME": str(self.work_dir),
            "PYTHONPATH": str(runner_root),
            "BACKENDAI_KERNEL_WORK_DIR": str(self.work_dir),
            "BACKENDAI_KERNEL_CONFIG_DIR": str(self.config_dir),
            "BACKENDAI_KERNEL_BIND_HOST": "127.0.0.1",
        }

    def _spawn(
        self, argv: Sequence[str], host_ports: Sequence[int], service_ports: Sequence[int]
    ) -> KernelProcessInfo:
        labels: dict[str, str] = {
            LabelName.AGENT_ID: str(self.agent_id),
            LabelName.KERNEL_ID: str(self.kernel_id),
            LabelName.SESSION_ID: str(self.session_id),
            LabelName.OWNER_USER: self.ownership_data.owner_user_id_to_str or "",
            LabelName.OWNER_PROJECT: self.ownership_data.owner_project_id_to_str or "",
            LabelName.OWNER_AGENT: str(self.agent_id),
            LabelName.KERNEL_SPEC: str(self.kspec_version),
        }
        (self.config_dir / "intrinsic-ports.json").write_bytes(
            dump_json({"replin": host_ports[0], "replout": host_ports[1]})
        )
        with (self.config_dir / LOG_FILENAME).open("ab") as log_file:
            # Not an asyncio subprocess: its transport kills the child when the agent exits.
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=self._runner_env(),
                cwd=self.work_dir,
                start_new_session=True,
            )
        threading.Thread(target=proc.wait, name=f"kernel-reaper-{proc.pid}", daemon=True).start()
        try:
            info = KernelProcessInfo(
                container_id=secrets.token_hex(32),
                pid=proc.pid,
                create_time=psutil.Process(proc.pid).create_time(),
                image=self.image_ref.canonical,
                labels=labels,
                host_ports=list(host_ports),
                service_ports=list(service_ports),
            )
            write_process_info(self.config_dir, info)
        except BaseException:
            # Without its record nothing could find this process again.
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            raise
        return info

    @override
    async def start_container(
        self,
        kernel_obj: AbstractKernel,
        cmdargs: list[str],
        resource_opts: Mapping[str, Any] | None,
        preopen_ports: Sequence[int],
        cluster_info: ClusterInfo,
    ) -> Mapping[str, Any]:
        if _CONTAINER_KRUNNER_PYTHON not in cmdargs:
            raise InvalidArgumentError(f"Unexpected kernel runner command: {cmdargs}")
        # Drops a jail prefix as well: there is no sandbox to run it in.
        argv = [sys.executable, *cmdargs[cmdargs.index(_CONTAINER_KRUNNER_PYTHON) + 1 :]]
        # No NAT in front of a host process: the declared port is the host port.
        service_ports = sorted({
            port for sport in kernel_obj.service_ports for port in sport["container_ports"]
        })
        # Held until the record is written, so that concurrent creations see each other's ports.
        async with self.service_port_lock:
            conflicts = await run_in_executor_with_context(
                None,
                find_port_conflicts,
                self.local_config.container.scratch_root,
                self.kernel_id,
                service_ports,
            )
            if conflicts:
                raise PortConflictError(
                    f"Service ports already in use on the agent host: {conflicts}"
                )
            host_ports = [self.port_pool.acquire() for _ in self.repl_ports]
            try:
                info = await run_in_executor_with_context(
                    None, self._spawn, argv, host_ports, service_ports
                )
            except Exception:
                self.port_pool.release_many(host_ports)
                raise
        for sport in kernel_obj.service_ports:
            sport["host_ports"] = tuple(sport["container_ports"])
        container_config = self.local_config.container
        return {
            "container_id": info.container_id,
            "kernel_host": container_config.advertised_host or container_config.bind_host,
            "repl_in_port": host_ports[0],
            "repl_out_port": host_ports[1],
            "stdin_port": 0,  # legacy
            "stdout_port": 0,  # legacy
            "host_ports": host_ports,
            "domain_socket_proxies": [],
            "block_service_ports": self.internal_data.get("block_service_ports", False),
        }


class NativeAgent(AbstractAgent[NativeKernel, NativeKernelCreationContext]):
    _service_port_lock: asyncio.Lock
    _registry_loader: PickleBasedKernelRegistryLoader
    _registry_writer: PickleBasedKernelRegistryWriter

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._service_port_lock = asyncio.Lock()
        agent_config = self.local_config.agent
        registry_file_name = f"last_registry.{self.id}.dat"
        registry_file_path = agent_config.var_base_path / registry_file_name
        self._registry_loader = PickleBasedKernelRegistryLoader(
            registry_file_path, agent_config.ipc_base_path / registry_file_name
        )
        self._registry_writer = PickleBasedKernelRegistryWriter(registry_file_path)

    @override
    async def __ainit__(self) -> None:
        log.info(
            "native backend selected without isolation or resource enforcement",
            kernel_os_user=psutil.Process().username(),
        )
        await super().__ainit__()

    @property
    def _scratch_root(self) -> Path:
        return self.local_config.container.scratch_root

    async def _read_process_info(self, kernel_id: KernelId) -> KernelProcessInfo | None:
        return await run_in_executor_with_context(
            None, read_process_info, config_dir_of(self._scratch_root, kernel_id)
        )

    @override
    async def _load_kernel_registry_from_recovery(self) -> MutableMapping[KernelId, AbstractKernel]:
        try:
            return await self._registry_loader.load_kernel_registry()
        except (KernelRegistryNotFound, KernelRegistryLoadError):
            # First start, or an unreadable file: kernels still running get terminated.
            return {}

    @override
    async def _write_kernel_registry_to_recovery(
        self,
        kernel_registry: MutableMapping[KernelId, AbstractKernel],
        metadata: KernelRegistrySaveMetadata,
    ) -> None:
        # Always forced: a kernel missing from the file is killed at the next agent start.
        await self._registry_writer.save_kernel_registry(
            kernel_registry, KernelRegistrySaveMetadata(force=True)
        )

    @override
    async def enumerate_containers(
        self,
        status_filter: frozenset[ContainerStatus] = ACTIVE_STATUS_SET,
    ) -> Sequence[tuple[KernelId, Container]]:
        def _scan() -> list[tuple[KernelId, Container]]:
            result = []
            for info_path in self._scratch_root.glob(f"*/config/{PROCESS_INFO_FILENAME}"):
                info = read_process_info(info_path.parent)
                if info is None or info.labels.get(LabelName.OWNER_AGENT) != str(self.id):
                    continue
                container = info.to_container(info_path.parent.resolve())
                if container.status in status_filter:
                    result.append((container.kernel_id, container))
            return result

        return await run_in_executor_with_context(None, _scan)

    @override
    async def resolve_image_distro(self, image: ImageConfig) -> str:
        # Only selects container-side runner binaries, which a host process does not use.
        return image["labels"].get(LabelName.BASE_DISTRO) or sys.platform

    @override
    async def get_cgroup_path(
        self, controller: CgroupController, container_id: ContainerId
    ) -> Path:
        return Path()

    @override
    def get_cgroup_version(self) -> str:
        return ""

    @override
    async def extract_image_command(self, image: str) -> list[str] | None:
        return None

    @override
    async def scan_images(self) -> ScanImagesResult:
        # Images are metadata only; whether one is usable is checked at kernel creation.
        return ScanImagesResult(scanned_images={}, removed_images={})

    @override
    async def pull_image(
        self,
        image_ref: ImageRef,
        registry_conf: ImageRegistry,
        *,
        timeout_seconds: float | None,
    ) -> None:
        pass

    @override
    async def push_image(
        self,
        image_ref: ImageRef,
        registry_conf: ImageRegistry,
        *,
        timeout_seconds: float | None | Sentinel = Sentinel.TOKEN,
    ) -> None:
        raise UnsupportedBackendOperationError("A host-process kernel has no image to push.")

    @override
    async def purge_images(self, request: PurgeImagesReq) -> PurgeImagesResp:
        return PurgeImagesResp([])

    @override
    async def check_image(
        self, image_ref: ImageRef, image_id: str, auto_pull: AutoPullBehavior
    ) -> bool:
        return False

    @override
    async def init_kernel_context(
        self,
        ownership_data: KernelOwnershipData,
        kernel_image: ImageRef,
        kernel_config: KernelCreationConfig,
        *,
        restarting: bool = False,
        cluster_ssh_port_mapping: ClusterSSHPortMapping | None = None,
    ) -> NativeKernelCreationContext:
        distro = await self.resolve_image_distro(kernel_config["image"])
        return NativeKernelCreationContext(
            ownership_data,
            self.event_producer,
            kernel_image,
            kernel_config,
            distro,
            self.local_config,
            self.computers,
            self.port_pool,
            self._service_port_lock,
            restarting=restarting,
        )

    @override
    async def destroy_kernel(
        self,
        kernel_id: KernelId,
        container_id: ContainerId | None,
    ) -> None:
        info = await self._read_process_info(kernel_id)
        if info is not None:
            await self._mark_exit_handled(kernel_id, info)
            await terminate_process_group(info, _TERMINATION_GRACE_PERIOD)

    async def _mark_exit_handled(self, kernel_id: KernelId, info: KernelProcessInfo) -> None:
        info.exit_handled = True
        await run_in_executor_with_context(
            None, write_process_info, config_dir_of(self._scratch_root, kernel_id), info
        )

    @override
    async def clean_kernel(
        self,
        kernel_id: KernelId,
        container_id: ContainerId | None,
        restarting: bool,
    ) -> None:
        config_dir = config_dir_of(self._scratch_root, kernel_id)
        info = await self._read_process_info(kernel_id)
        if info is not None and not info.exit_handled and not restarting:
            # No destroy preceded this clean: the runner died on its own. Without a result
            # event the manager would end the session with its result undefined.
            await self._mark_exit_handled(kernel_id, info)
            session_id = SessionId(UUID(info.labels[LabelName.SESSION_ID]))
            reason = KernelLifecycleEventReason.SELF_TERMINATED
            await self.anycast_and_broadcast_event(
                SessionFailureAnycastEvent(session_id=session_id, reason=reason),
                SessionFailureBroadcastEvent(session_id=session_id, reason=reason),
            )
        if info is not None:
            # Children may outlive a runner that died on its own.
            await terminate_process_group(info, _TERMINATION_GRACE_PERIOD)
            if container_id is not None:

                async def log_iter() -> AsyncGenerator[bytes, None]:
                    # ponytail: whole log in memory; read in chunks if kernel logs grow large.
                    try:
                        yield await run_in_executor_with_context(
                            None, (config_dir / LOG_FILENAME).read_bytes
                        )
                    except FileNotFoundError:
                        return

                await self.collect_logs(kernel_id, container_id, log_iter())
        if not restarting:
            await run_in_executor_with_context(
                None, lambda: shutil.rmtree(config_dir.parent, ignore_errors=True)
            )

    @override
    async def create_local_network(self, network_name: str) -> None:
        pass

    @override
    async def destroy_local_network(self, network_name: str) -> None:
        pass

    @override
    async def restart_kernel__load_config(
        self,
        kernel_id: KernelId,
        name: str,
    ) -> bytes:
        config_dir = config_dir_of(self._scratch_root, kernel_id)
        return await run_in_executor_with_context(None, (config_dir / name).read_bytes)

    @override
    async def restart_kernel__store_config(
        self,
        kernel_id: KernelId,
        name: str,
        data: bytes,
    ) -> None:
        config_dir = config_dir_of(self._scratch_root, kernel_id)
        await run_in_executor_with_context(None, (config_dir / name).write_bytes, data)
