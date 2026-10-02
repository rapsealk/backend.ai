from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ai.backend.kernel as kernel_pkg
from ai.backend.kernel.app import Runner
from ai.backend.kernel.utils import get_config_dir, get_work_dir, scan_proc_stats

DEFAULT_SOCK_PATH = "/tmp/bai-user-input.sock"


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch dir laid out like the agent's; the cwd is restored after the test."""
    (tmp_path / "work").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "environ.txt").write_text("FROM_ENVIRON_TXT=yes\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FROM_ENVIRON_TXT", raising=False)
    return tmp_path


@pytest.fixture
def host_env(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BACKENDAI_KERNEL_WORK_DIR", str(scratch / "work"))
    monkeypatch.setenv("BACKENDAI_KERNEL_CONFIG_DIR", str(scratch / "config"))
    return scratch


@pytest.fixture
def container_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKENDAI_KERNEL_WORK_DIR", raising=False)
    monkeypatch.delenv("BACKENDAI_KERNEL_CONFIG_DIR", raising=False)


class TestKernelDirs:
    def test_defaults_are_container_paths(self, container_env: None) -> None:
        assert get_work_dir() == Path("/home/work")
        assert get_config_dir() == Path("/home/config")

    def test_overridden_by_environment(self, host_env: Path) -> None:
        assert get_work_dir() == host_env / "work"
        assert get_config_dir() == host_env / "config"


class TestRunnerAsHostProcess:
    def test_uses_overridden_dirs(self, host_env: Path) -> None:
        work_dir = host_env / "work"

        runner = Runner(Path("/bin/true"))

        assert runner.child_env["HOME"] == str(work_dir)
        assert runner.child_env["PATH"].split(":")[0] == str(work_dir / ".local/bin")
        assert runner.child_env["FROM_ENVIRON_TXT"] == "yes"
        assert Path.cwd() == work_dir.resolve()

    def test_user_input_socket_is_per_process(self, host_env: Path) -> None:
        runner = Runner(Path("/bin/true"))

        assert str(os.getpid()) in runner.user_input_sock_path
        assert runner.child_env["_BACKEND_USER_INPUT_SOCK"] == runner.user_input_sock_path


class TestRunnerInContainer:
    @pytest.fixture
    def container_dirs(self, scratch: Path, container_env: None) -> Generator[Path, None, None]:
        """Stand-ins for /home/work and /home/config, which a test host does not have."""
        with (
            patch("ai.backend.kernel.base.get_work_dir", return_value=scratch / "work"),
            patch("ai.backend.kernel.base.get_config_dir", return_value=scratch / "config"),
        ):
            yield scratch

    def test_keeps_cwd_and_shared_socket(self, container_dirs: Path) -> None:
        runner = Runner(Path("/bin/true"))

        assert Path.cwd() == container_dirs.resolve()
        assert runner.user_input_sock_path == DEFAULT_SOCK_PATH
        assert runner.child_env["_BACKEND_USER_INPUT_SOCK"] == DEFAULT_SOCK_PATH


class TestSshdInit:
    @pytest.fixture
    def runner(self, host_env: Path) -> Runner:
        runner = Runner(Path("/bin/true"))
        runner.init_done = asyncio.Event()
        return runner

    @pytest.fixture
    def mock_init_sshd(self) -> Generator[AsyncMock, None, None]:
        with patch("ai.backend.kernel.base.init_sshd_service", new_callable=AsyncMock) as mock:
            yield mock

    async def test_skipped_without_dropbear(
        self, runner: Runner, mock_init_sshd: AsyncMock, host_env: Path
    ) -> None:
        with patch("ai.backend.kernel.base.DROPBEAR_PATH", host_env / "missing"):
            await runner._init_with_loop()

        mock_init_sshd.assert_not_awaited()

    async def test_runs_with_dropbear(
        self, runner: Runner, mock_init_sshd: AsyncMock, host_env: Path
    ) -> None:
        with patch("ai.backend.kernel.base.DROPBEAR_PATH", host_env / "config" / "environ.txt"):
            await runner._init_with_loop()

        mock_init_sshd.assert_awaited_once()


class TestScanProcStats:
    def test_empty_without_procfs(self) -> None:
        with patch.object(Path, "is_dir", return_value=False):
            assert scan_proc_stats() == {}


class TestReexec:
    @pytest.fixture
    def mock_execvpe(self, monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock, None, None]:
        monkeypatch.delenv("BACKEND_REEXECED", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("PYTHONHOME", raising=False)
        with patch("os.execvpe") as mock:
            yield mock
        Path(f"/tmp/{os.getpid()}.krunner-args").unlink(missing_ok=True)

    def test_venv_interpreter_gets_no_pythonhome(
        self, mock_execvpe: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "prefix", "/venv")
        monkeypatch.setattr(sys, "base_prefix", "/usr")

        kernel_pkg._reexec_with_argv0("bai-krunner")

        assert "PYTHONHOME" not in mock_execvpe.call_args.args[2]

    def test_standalone_interpreter_gets_pythonhome(
        self, mock_execvpe: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "prefix", "/opt/backend.ai")
        monkeypatch.setattr(sys, "base_prefix", "/opt/backend.ai")

        kernel_pkg._reexec_with_argv0("bai-krunner")

        assert mock_execvpe.call_args.args[2]["PYTHONHOME"] == "/opt/backend.ai"
