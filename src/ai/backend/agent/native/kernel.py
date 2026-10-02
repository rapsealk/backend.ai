from __future__ import annotations

import io
import json
import os
import stat
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, override

from ai.backend.agent.errors import KernelRunnerNotInitializedError
from ai.backend.agent.errors.backend import UnsupportedBackendOperationError
from ai.backend.agent.kernel import AbstractCodeRunner, AbstractKernel
from ai.backend.agent.types import AgentEventData
from ai.backend.common.asyncio import run_in_executor_with_context
from ai.backend.common.dto.agent.response import CodeCompletionResp
from ai.backend.common.events.dispatcher import EventProducer
from ai.backend.common.types import CommitStatus, KernelId, SessionId

from .process import LOG_FILENAME, host_path_of, rebase_kernel_path

_MAX_DOWNLOAD_SIZE: Final = 1048576  # 1 MiB, as in the Docker backend


class NativeKernel(AbstractKernel):
    @property
    def _scratch_dir(self) -> Path:
        scratch_root: Path = self.agent_config["container"]["scratch-root"]
        return (scratch_root / str(self.kernel_id)).resolve()

    @property
    @override
    def stats_enabled(self) -> bool:
        # ponytail: no per-kernel stats; walk the process tree with psutil when they are needed.
        return False

    @override
    async def close(self) -> None:
        pass

    def _runner(self) -> AbstractCodeRunner:
        if self.runner is None:
            raise KernelRunnerNotInitializedError("Kernel runner is not initialized")
        return self.runner

    @override
    async def create_code_runner(
        self, event_producer: EventProducer, *, client_features: frozenset[str], api_version: int
    ) -> AbstractCodeRunner:
        return await NativeCodeRunner.new(
            self.kernel_id,
            self.session_id,
            event_producer,
            repl_in_port=self.data["repl_in_port"],
            repl_out_port=self.data["repl_out_port"],
            exec_timeout=0,
            client_features=client_features,
        )

    @override
    async def check_status(self) -> dict[str, Any] | None:
        return await self._runner().feed_and_get_status()

    @override
    async def get_completions(self, text: str, opts: Mapping[str, Any]) -> CodeCompletionResp:
        return CodeCompletionResp(result=await self._runner().feed_and_get_completion(text, opts))

    @override
    async def get_logs(self) -> dict[str, Any]:
        log_path = self._scratch_dir / "config" / LOG_FILENAME
        try:
            logs = await run_in_executor_with_context(None, log_path.read_bytes)
        except FileNotFoundError:
            logs = b""
        return {"logs": logs.decode("utf-8", "replace")}

    @override
    async def interrupt_kernel(self) -> dict[str, Any]:
        await self._runner().feed_interrupt()
        return {"status": "finished"}

    @override
    async def start_service(self, service: str, opts: Mapping[str, Any]) -> dict[str, Any]:
        if self.data.get("block_service_ports", False):
            return {"status": "failed", "error": "operation blocked"}
        for sport in self.service_ports:
            if sport["name"] == service:
                break
        else:
            return {"status": "failed", "error": "invalid service name"}
        return await self._runner().feed_start_service({
            "name": service,
            "port": sport["container_ports"][0],  # primary port
            "ports": sport["container_ports"],
            "protocol": sport["protocol"],
            "options": opts,
        })

    @override
    async def start_model_service(self, model_service: Mapping[str, Any]) -> dict[str, Any]:
        # The manager resolved the definition against kernel-side paths; point it at the host.
        kernel_path: str = model_service["model_path"]
        host_path = str(host_path_of(self._scratch_dir, kernel_path))
        model_service = {**model_service, "model_path": host_path}
        service = model_service.get("service")
        if service and service.get("start_command"):
            start_command = rebase_kernel_path(service["start_command"], kernel_path, host_path)
            model_service["service"] = {**service, "start_command": start_command}
        return await self._runner().feed_start_model_service(model_service)

    @override
    async def shutdown_service(self, service: str) -> None:
        await self._runner().feed_shutdown_service(service)

    @override
    async def get_service_apps(self) -> dict[str, Any]:
        return await self._runner().feed_service_apps()

    @override
    async def check_duplicate_commit(self, kernel_id: KernelId, subdir: str) -> CommitStatus:
        return CommitStatus.READY

    @override
    async def commit(
        self,
        kernel_id: KernelId,
        subdir: str,
        *,
        canonical: str | None = None,
        filename: str | None = None,
        extra_labels: dict[str, str] | None = None,
    ) -> None:
        raise UnsupportedBackendOperationError("A host-process kernel has no image to commit.")

    def _resolve_in_work_dir(self, path: os.PathLike[str] | str) -> Path:
        # vfolders are symlinks pointing outside the work dir, so normalize without resolving.
        work_dir = self._scratch_dir / "work"
        abspath = Path(os.path.normpath(work_dir / path))
        if not abspath.is_relative_to(work_dir):
            raise PermissionError("Not allowed to access files outside the kernel work directory")
        return abspath

    @override
    async def accept_file(self, container_path: os.PathLike[str] | str, filedata: bytes) -> None:
        host_abspath = self._resolve_in_work_dir(container_path)

        def _write_to_disk() -> None:
            host_abspath.parent.mkdir(parents=True, exist_ok=True)
            host_abspath.write_bytes(filedata)

        await run_in_executor_with_context(None, _write_to_disk)

    @override
    async def download_file(self, container_path: os.PathLike[str] | str) -> bytes:
        host_abspath = self._resolve_in_work_dir(container_path)

        def _make_tar() -> bytes:
            with io.BytesIO() as buf:
                with tarfile.open(fileobj=buf, mode="w") as tar:
                    tar.add(host_abspath, arcname=host_abspath.name)
                return buf.getvalue()

        tarbytes = await run_in_executor_with_context(None, _make_tar)
        if len(tarbytes) > _MAX_DOWNLOAD_SIZE:
            raise ValueError("Too large archive file exceeding 1 MiB")
        return tarbytes

    @override
    async def download_single(self, container_path: os.PathLike[str] | str) -> bytes:
        host_abspath = self._resolve_in_work_dir(container_path)

        def _read() -> bytes:
            if host_abspath.stat().st_size > _MAX_DOWNLOAD_SIZE:
                raise ValueError("Too large file exceeding 1 MiB")
            return host_abspath.read_bytes()

        return await run_in_executor_with_context(None, _read)

    @override
    async def list_files(self, container_path: os.PathLike[str] | str) -> dict[str, Any]:
        host_abspath = self._resolve_in_work_dir(container_path)

        def _scan() -> list[dict[str, Any]]:
            files = []
            for f in os.scandir(host_abspath):
                fstat = f.stat(follow_symlinks=False)
                files.append({
                    "mode": stat.filemode(fstat.st_mode),
                    "size": fstat.st_size,
                    "ctime": fstat.st_ctime,
                    "mtime": fstat.st_mtime,
                    "atime": fstat.st_atime,
                    "filename": f.name,
                })
            return files

        try:
            files, errors = await run_in_executor_with_context(None, _scan), ""
        except OSError as e:
            files, errors = [], repr(e)
        return {"files": json.dumps(files), "errors": errors, "abspath": str(container_path)}

    @override
    async def notify_event(self, evdata: AgentEventData) -> None:
        await self._runner().feed_event(evdata)


class NativeCodeRunner(AbstractCodeRunner):
    repl_in_port: int
    repl_out_port: int

    def __init__(
        self,
        kernel_id: KernelId,
        session_id: SessionId,
        event_producer: EventProducer,
        *,
        repl_in_port: int,
        repl_out_port: int,
        exec_timeout: float = 0,
        client_features: frozenset[str] | None = None,
    ) -> None:
        super().__init__(
            kernel_id,
            session_id,
            event_producer,
            exec_timeout=exec_timeout,
            client_features=client_features,
        )
        self.repl_in_port = repl_in_port
        self.repl_out_port = repl_out_port

    # The runner binds its REPL sockets to the loopback only.
    @override
    async def get_repl_in_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.repl_in_port}"

    @override
    async def get_repl_out_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.repl_out_port}"
