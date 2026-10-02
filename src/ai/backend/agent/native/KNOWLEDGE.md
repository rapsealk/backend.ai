---
name: agent-native-backend
type: design-rationale
description: the native agent backend where a kernel is the kernel runner running as a host process tree, the on-disk kernel record as the kernel's identity, kernel-path to host-path mapping by symlink, identity service-port mapping and its conflict rule, images as metadata, death reporting through exit_handled, restart re-adoption through the pickle registry, absence of isolation and resource enforcement
scope: src/ai/backend/agent/native
keywords: [NativeAgent, NativeKernel, NativeKernelCreationContext, KernelProcessInfo, native-kernel.json, host_path_of, find_port_conflicts, exit_handled, Metal, BEP-1079]
sources:
  - src/ai/backend/agent/native/agent.py
  - src/ai/backend/agent/native/kernel.py
  - src/ai/backend/agent/native/process.py
  - src/ai/backend/agent/native/resources.py
  - src/ai/backend/kernel/utils.py
generated:
  by: claude-code/fable-5.1
  at: 2026-10-03
status: stable
---

# Agent native backend — Knowledge

> Rules: `AGENTS.md` in `src/ai/backend/agent`. Design and decision log: [BEP-1079](../../../../../proposals/BEP-1079-native-process-agent-backend.md).

## Why this package exists

`backend = "native"` runs a kernel as a host process tree instead of a container.
It exists because no Linux container on macOS reaches Metal, so a session that uses the Apple GPU has to run on the host.

## A kernel is the kernel runner as a host process

- `NativeKernelCreationContext` spawns `sys.executable -s -m ai.backend.kernel <runtime-type> <runtime-path>` in a new process session, so the runner's pid is also the process group id.
- The runner gets a minimal environment; `BACKENDAI_KERNEL_WORK_DIR`, `BACKENDAI_KERNEL_CONFIG_DIR` and `BACKENDAI_KERNEL_BIND_HOST` point it at the scratch directory and the loopback.
- Agent↔runner ZMQ ports come from the agent port pool and reach the runner through `intrinsic-ports.json`.
- `execute`, `start_service` and `start_model_service` go through the runner, as in a container.

## The on-disk record is the kernel's identity

- `config/native-kernel.json` (`KernelProcessInfo`) holds pid, process create time, labels, REPL ports, declared service ports and `exit_handled`.
- `enumerate_containers`, `destroy_kernel` and `clean_kernel` work from the record, not from the in-memory `NativeKernel`.
- pid plus create time identifies the process; a reused pid is treated as a dead kernel.
- The record is written through a temporary file and a rename, because a kernel with a truncated record cannot be found again.

## Kernel paths are mapped, not mounted

| Kernel path | Host path |
|---|---|
| `/home/work` | `<scratch-root>/<kernel-id>/work` |
| `/home/work/<p>` | `<scratch-root>/<kernel-id>/work/<p>` |
| any other `/<p>` | `<scratch-root>/<kernel-id>/mounts/<p>` |
| `/home/config` | `<scratch-root>/<kernel-id>/config` |

- A mount is a symlink at the mapped path; read-only is not enforced.
- `/` and `/home/work` themselves cannot be mount targets.
- An inference session's `model_path`, the same path inside its start command, and `BACKEND_MODEL_PATH` are rewritten to the host path.

## The declared port is the host port

- No NAT exists in front of a host process, so `host_ports` of every service port equals its `container_ports`.
- Creation fails with `PortConflictError` when another kernel's record declares the port, or something already listens on it on the loopback.
- The check and the record write run under one agent-level lock, so concurrent creations see each other.
- `sshd` and `ttyd` are removed from the service-port list: they need runner binaries that exist only in a container.

## An image is metadata

- The image row supplies labels; nothing is pulled, scanned, pushed or committed.
- `ai.backend.runtime-path` must be an executable on the agent host, else `KernelRuntimeNotFoundError`; its directory is prepended to `PATH`.
- The agent does not report installed images; a wrong runtime path surfaces at kernel creation.

## A clean without a destroy is a death

- `destroy_kernel` sets `exit_handled` before it signals the process group.
- `clean_kernel` that finds `exit_handled` unset emits `SessionFailure` with reason `self-terminated`; without it the manager would end the session with an undefined result.
- The flag is on disk, so a kernel that died while the agent was down is reported when the agent starts.

## Restart re-adopts through the pickle registry

- `NativeAgent` saves and loads the kernel registry with the pickle loader and writer the Kubernetes backend uses; saves are always forced.
- With the registry loaded, the base agent restores ports, slot allocations and the code runner from the records.
- A running kernel missing from the registry is terminated at agent start, like a container of an unregistered kernel.
- The kernel survives an agent stop because it was started in its own process session and not as an asyncio subprocess.

## Nothing is isolated or enforced

| Aspect | State |
|---|---|
| OS user | The agent's own |
| Filesystem, network, devices | Whatever that user can reach |
| `cpu`, `mem`, accelerator slots | Accounted through `alloc_map`, not limited |
| Per-kernel statistics | None |

- Selecting the backend is the opt-in; the agent states it in its startup log.
- The supported use is a single-tenant development agent.
