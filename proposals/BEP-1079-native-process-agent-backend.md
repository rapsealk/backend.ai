---
Author: Jeongseok Kang (jskang@lablup.com)
Status: Draft
Created: 2026-10-02
Created-Version: 26.9.0
Target-Version:
Implemented-Version:
---

<!-- context-for-ai
type: master-bep
scope: Add a fourth agent backend, `native`, where a kernel is the kernel runner running as a host process tree, so a session on a macOS agent reaches the Apple GPU (Metal)
detail-docs: []
key-constraints:
  - No isolation and no resource enforcement; single-tenant development agents only
  - Built on the existing AbstractAgent / AbstractKernel / AbstractKernelCreationContext generics; all backend-specific code stays in agent/native/
  - Container behavior of the kernel runner is unchanged when its three variables are unset
  - No manager or scheduler change in the first cut
key-decisions:
  - Backend name `native` (AgentBackend.NATIVE), not built on BEP-1057 ComputeBackend yet
  - The kernel runner is reused on the host through BACKENDAI_KERNEL_WORK_DIR, BACKENDAI_KERNEL_CONFIG_DIR and BACKENDAI_KERNEL_BIND_HOST
  - Kernel paths map to the scratch directory by symlink; the declared service port is the host port
  - Running kernels are re-adopted after an agent restart
  - Serving runtime variant is mlx-lm; mlxcel v0.7.0 is deferred
  - An image row is metadata only; ai.backend.runtime-path is a host interpreter path
  - Slot `metal.device` (count, one device per host); overlap with `mem` is documented, not solved
phases: 8
-->

# Native Process Agent Backend

## Related Issues

- Epics: rapsealk/backend.ai#1, rapsealk/backend.ai#8 (sub-issues #2 – #7, #9, #10)
- Serving runtime evaluation: rapsealk/backend.ai#21
- Consumer: lablup/backend.ai-fasttrack#6041
- Related BEPs: [BEP-1002](BEP-1002-agent-architecture.md), [BEP-1016](BEP-1016-accelerator-interface-v2.md), [BEP-1057](BEP-1057-agent-re-architecture.md)

## 1. Goal

A Backend.AI session on a macOS agent uses the Apple GPU. Done when `mlx.core.default_device()` reports `Device(gpu, 0)` inside a session.

This BEP defines:

| Item | Content |
|---|---|
| Agent backend `native` | A kernel is a host process tree, not a container |
| Kernel runner path contract | Three environment variables that let `ai.backend.kernel` run outside a container |
| Compute plugin `metal` | The Apple GPU as the slot `metal.device` |

Out of scope:

- Multi-tenant isolation on macOS.
- Resource enforcement for `cpu`, `mem`, `metal.device`.
- GPU access from Linux containers through Vulkan.
- macOS guest VMs as a backend.

## 2. Motivation

No Linux container on macOS reaches Metal. A workload that needs the Apple GPU must run as a host process.

| Option | Metal / MLX GPU | Evidence |
|---|---|---|
| Docker Desktop, OrbStack container | None | `--gpus` is Windows/WSL2 only ([docker docs](https://github.com/docker/docs/blob/main/content/manuals/desktop/features/gpu.md)); Docker Model Runner runs its engines as host processes under `sandbox-exec` ([docs](https://docs.docker.com/ai/model-runner/)); OrbStack GPU request is open ([orbstack#1818](https://github.com/orbstack/orbstack/issues/1818)) |
| Podman libkrun (`--device /dev/dri`) | Vulkan compute only | Works for llama.cpp ([podman docs](https://podman-desktop.io/docs/podman/gpu)); MLX has no Vulkan backend ([mlx#1751](https://github.com/ml-explore/mlx/issues/1751)) |
| Apple `container` | None | No GPU ([container#1511](https://github.com/apple/container/issues/1511)), no Docker API ([container#66](https://github.com/apple/container/issues/66)) |
| macOS guest VM | Paravirtual Metal | macOS SLA: two VMs per host, development and testing use ([SLA](https://www.apple.com/legal/sla/docs/macOSTahoe.pdf)) |
| Host process | Full | Ray advertises the Apple GPU to host worker processes ([ray#38464](https://github.com/ray-project/ray/pull/38464)) |

The MLX maintainers state the container limit directly: [mlx#1945](https://github.com/ml-explore/mlx/issues/1945).

Measured on an Apple M5 Pro (48 GB, macOS 27.0), `mlx_lm lora` (20 iterations) → `fuse` → `server` → `POST /v1/chat/completions` with `HuggingFaceTB/SmolLM2-135M-Instruct`:

| Where | MLX device | Train step wall time |
|---|---|---|
| Host process | `Device(gpu, 0)` | 7 s |
| `linux/arm64` container, `mlx[cpu]` | `Device(cpu, 0)` | 266 s |

The container figure includes model and dataset download; the host run had a warm cache.

## 3. Current State & Scope, by Area

Per area, **✅ exists / ➕ to add**.

### 3.1 Agent backend

| | Item |
|---|---|
| ✅ | `AgentBackend` is `docker` / `kubernetes` / `dummy`; the agent loads `ai.backend.agent.{backend}` by name |
| ✅ | The Docker backend runs on darwin, with Linux containers |
| ✅ | BEP-1057 `ComputeBackend`: the ABC and `DummyComputeBackend` have landed; Docker is not migrated |
| ➕ | `AgentBackend.NATIVE` and the package `ai.backend.agent.native` (4.1) |

### 3.2 Kernel runner

| | Item |
|---|---|
| ✅ | `ai.backend.kernel` is launched in the container as `/opt/backend.ai/bin/python -s -m ai.backend.kernel <runtime-type> <runtime-path>` |
| ✅ | The runner reads its ZMQ ports from `intrinsic-ports.json` (`replin`, `replout`) when the file exists |
| ✅ | `/home/work`, `/home/config`, `prctl`, `/opt/kernel/*` binaries and `LD_PRELOAD` are assumed |
| ➕ | Work and config directories taken from the environment; Linux-only parts skipped on darwin (4.3) |

### 3.3 Image and scheduling

| | Item |
|---|---|
| ✅ | A session requires an image row; the `local` registry scans the Docker daemon on the manager host |
| ✅ | The scheduler matches an agent to an image by `architecture` only |
| ➕ | An image row used as metadata for a host environment (4.5) |
| ➕ | No scheduler change. Native agents are placed in a dedicated resource group (Open Questions) |

### 3.4 Compute plugin

| | Item |
|---|---|
| ✅ | `AbstractComputePlugin` takes Docker objects in `generate_docker_args()` and `restore_from_container()` |
| ✅ | BEP-1016 (Draft) defines a workload that is a container or a process tree; not implemented |
| ➕ | `ai.backend.accelerator.metal` on the current `AbstractComputePlugin`; the Docker-shaped methods return nothing (4.7) |

### 3.5 Model serving

| | Item |
|---|---|
| ✅ | Runtime variants `vllm`, `nim`, `cmd`, `custom`, `huggingface-tgi`, `sglang`, `modular-max`, `llama-cpp` |
| ➕ | Runtime variant `mlx-lm` that starts `mlx_lm.server` |

## 4. Proposed Design

### 4.1 Backend selection

```toml
[agent]
backend = "native"
allow-compute-plugins = ["ai.backend.accelerator.metal"]
```

- `ai.backend.agent.native` provides `NativeAgent`, `NativeKernel`, `NativeKernelCreationContext` on the existing generics.
- The backend adds no abstract method to `AbstractAgent`, `AbstractKernel`, or `AbstractKernelCreationContext`.
- Selecting `backend = "native"` is the opt-in to running without isolation. The agent logs it at startup.

### 4.2 Kernel process

```
Manager ──RPC──▶ NativeAgent ──spawn──▶ kernel runner (host process, own process session)
                     │                        │
                     └──── ZMQ, host ports ───┘──▶ user processes (batch command, services)
```

| Aspect | Contract |
|---|---|
| Command | `sys.executable -s -m ai.backend.kernel <runtime-type> <runtime-path>`, the agent's own interpreter |
| Process session | The runner starts in a new process session; every user process belongs to its process group |
| Agent ↔ runner | ZMQ over host ports taken from the agent port pool, written to `intrinsic-ports.json` as `replin` / `replout`. The runner binds them on the loopback and the agent connects there |
| OS user | The agent's OS user |
| RPC surface | Unchanged: `execute`, `start_service`, `start_model_service` go through the runner as in a container |

### 4.3 Kernel runner path contract

| Variable | Default | Meaning |
|---|---|---|
| `BACKENDAI_KERNEL_WORK_DIR` | `/home/work` | `HOME` and working directory of user processes |
| `BACKENDAI_KERNEL_CONFIG_DIR` | `/home/config` | Location of `environ.txt`, `intrinsic-ports.json` |
| `BACKENDAI_KERNEL_BIND_HOST` | `*` | Bind address of the runner's ZMQ sockets; the `native` backend passes `127.0.0.1` |

- With all three unset, the runner behaves as it does in a container today.
- On darwin, `prctl`, `/opt/kernel/*` binaries, `/opt/backend.ai`, `LD_PRELOAD`, and the sshd and ttyd intrinsic services are skipped. Their absence does not stop the runner.

### 4.4 Filesystem

| Kernel path | Host path |
|---|---|
| `/home/work` | `<scratch-root>/<kernel-id>/work`, the `HOME` and working directory |
| `/home/work/<path>` | `<scratch-root>/<kernel-id>/work/<path>` |
| Any other `/<path>` | `<scratch-root>/<kernel-id>/mounts/<path>` |
| `/home/config` | `<scratch-root>/<kernel-id>/config` |

- A mount is a symlink at the mapped path to the vfolder host path. `/` and `/home/work` are refused as mount targets.
- For an inference session, `model_path`, that path inside the start command, and `BACKEND_MODEL_PATH` are rewritten to the host path. `pre_start_actions` arguments are not.

Files the backend adds:

| File | Content |
|---|---|
| `config/native-kernel.json` | pid, process create time, labels, REPL ports, declared service ports, `exit_handled` |
| `config/kernel.log` | Output of the runner and the user processes |
| `<var-base-path>/last_registry.<agent-id>.dat` | The agent's kernel registry |

Limits:

- `/home/work/...` as an absolute path does not exist. Workloads address vfolders relative to `HOME`.
- A read-only vfolder mount is not enforced.
- The process sees the whole host filesystem the agent's OS user can read.

### 4.5 Image

An image row supplies labels and nothing else. Nothing is pulled.

| Label | Use in the `native` backend |
|---|---|
| `ai.backend.runtime-type` | Selects the runner class, as in a container |
| `ai.backend.runtime-path` | Absolute host path of the interpreter; its directory is prepended to `PATH` |
| `ai.backend.service-ports` and the remaining labels | Same meaning as in a container |

Registration in the first cut: a Docker image that carries the labels and no filesystem content, scanned through the `local` registry. `architecture` of the row is the agent's (`aarch64`).

The agent does not report installed images. It checks the runtime path at kernel creation; a path that is not an executable on the host fails the creation.

### 4.6 Ports

- No NAT. The host port of a service equals the port the image label, the preopen list, or the model definition declares.
- Kernel creation fails with `PortConflictError` when another kernel's `native-kernel.json` declares the port, or a listener answers on it on the loopback.
- `sshd` and `ttyd` are not advertised: their binaries exist only in a container.
- Precedent: the host-network branch in `agent/docker/intrinsic.py` that writes `intrinsic-ports.json`.

### 4.7 Resources and the `metal.device` slot

| Slot | Accounting | Enforcement |
|---|---|---|
| `cpu`, `mem` | `alloc_map`, as in other backends | None (no cgroups) |
| `metal.device` | `DiscretePropertyAllocMap`, count | None |

Compute plugin `ai.backend.accelerator.metal`:

| Aspect | Contract |
|---|---|
| Key / slot | `metal` / `metal.device` |
| Devices | One per host |
| Device memory | Metal `recommendedMaxWorkingSetSize` (40.2 GB on a 48 GB host) |
| Discovery and node stats | `ioreg -r -d 1 -w 0 -c IOAccelerator`, `sysctl hw.memsize`, `system_profiler SPDisplaysDataType -json`; none needs sudo |
| Off macOS | The plugin reports no device |

- Device memory is the same RAM that `mem` accounts. A session holding `metal.device` and `mem` can be counted twice against physical memory.
- Any host process can open the Metal device. `metal.device` schedules and accounts; it does not gate access.
- The manager reports the slot once the slot type `metal.device` is registered (`fixtures/manager/example-resource-slot-types.json`, or `resource-slot slot-type create` on an existing install).
- `[resource] allocation-order` in the agent config lists `metal`.

### 4.8 Isolation

None. The kernel runs as the agent's OS user with that user's filesystem, network, and device access.

| Stance | Content |
|---|---|
| Supported use | Single-tenant development agents |
| Opt-in | `backend = "native"` in the agent config |
| Not provided | Namespaces, seccomp, jail, cgroups |

Prior art for host-process engines on macOS:

- Docker Model Runner had two container-to-host CVEs in its host-process engines in 2026 (CVE-2026-5843, CVE-2026-5817, [release notes](https://github.com/docker/docs/blob/main/content/manuals/desktop/release-notes.md)).
- `sandbox-exec`, which Docker Model Runner uses, is marked deprecated in its macOS 27.0 man page.

### 4.9 Lifecycle

| Event | Behavior |
|---|---|
| Create | Allocate slots and ports, prepare the scratch directory and symlinks, spawn the runner, wait for its status reply |
| Destroy | `SIGTERM` to the process group, `SIGKILL` after 10 s |
| Clean | Release ports and slots, remove the scratch directory |
| Liveness | By pid and process create time from `native-kernel.json`. A runner that died without a destroy ends the session with result `FAILURE` and reason `self-terminated` |
| Agent stop | Kernels keep running |
| Agent restart | Running kernels are re-adopted: registry entry, slot allocations, ports, code runner. A kernel that died meanwhile is cleaned and its session fails with `self-terminated`. With a missing or unreadable registry file, running kernels are terminated |

## 5. Relation to Other BEPs

| BEP | Relation |
|---|---|
| BEP-1057 | `native` is a candidate `ComputeBackend` impl (BEP-1057 lists `vm/` as a future impl; a process tree is a third kind). It moves there after Phase 1 migrates Docker. Keeping the code in `agent/native/` bounds that move |
| BEP-1016 | `Workload.type = "process_tree"` and `AbstractLifecycleHook` are the plugin API this backend needs. The `metal` plugin is written on the current API and adopts BEP-1016 when it lands |
| BEP-1002 | The kernel-runner / Provisioner / Stage model is unchanged. `stage/kernel_lifecycle/` has Docker stages only; `native` does not add stages in the first cut |

## 6. Migration / Compatibility

- No change for `docker`, `kubernetes`, `dummy` agents.
- The kernel runner in a container reads no new required input; the three variables default to the current behavior.
- The kernel runner's shutdown waits at most 1 s for its tasks, in a container as well.
- No manager, scheduler, or DB schema change. The `mlx-lm` runtime variant is an additive seed.
- No RPC contract change between manager and agent.

## 7. Implementation Plan

One pull request per row, stacked in this order. All eight are implemented.

| # | Work | Verified by |
|---|---|---|
| 1 | This BEP | — |
| 2 | Kernel runner path contract (4.3) | The runner executes a batch command as a host process on macOS |
| 3 | `native` backend, batch session (4.1, 4.2, 4.4, 4.5, 4.8, 4.9) | A batch session prints `Device(gpu, 0)`; termination leaves no process |
| 4 | `metal` compute plugin (4.7) | A session is created with `metal.device = 1`; the agent reports GPU utilization and memory in node stats |
| 5 | `native` backend, service ports and inference sessions (4.4, 4.6) | An inference session serves a model folder through the deployment endpoint |
| 6 | `mlx-lm` runtime variant | A deployment answers `/v1/chat/completions` from `mlx_lm.server` on Metal |
| 7 | `native` backend, liveness and restart recovery (4.9) | A killed runner fails the session; a deployment keeps answering across an agent restart |
| 8 | Setup document and sample configs (`docs/agent/native.rst`) | Following it reaches the Goal |

## Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| 2026-10-02 | Build `native` on the `AbstractAgent` generics, confined to `agent/native/` | BEP-1057 `ComputeBackend` has the ABC and `DummyComputeBackend` only; services that would drive it are not extracted | Wait for BEP-1057 Phase 1 (blocks the Goal on an open-ended refactor); implement `ComputeBackend` now (nothing calls it) |
| 2026-10-02 | Do not wait for BEP-1016; write the `metal` plugin on the current `AbstractComputePlugin` | The plugin needs discovery, slots, and node stats only; the Docker-shaped methods have nothing to return for a host process | Implement BEP-1016 first (Draft, no implementation, larger than this BEP) |
| 2026-10-02 | An image row is metadata; `ai.backend.runtime-path` is a host interpreter path; registered through the `local` registry. No OS dimension in scheduling | Needs no manager change. Session creation, labels, and service ports keep working | New registry type for host environments; agent config mapping image → environment; `architecture` value `darwin-aarch64` |
| 2026-10-02 | Slot `metal.device`, count, one device per host, memory = `recommendedMaxWorkingSetSize`; overlap with `mem` documented | Matches the `{key}.device` convention of other plugins. A count slot gives exclusive scheduling of the one GPU | Memory-based slot (`metal.mem`) deducted from `mem`; no slot (GPU unaccounted) |
| 2026-10-02 | Reuse the kernel runner on the host through `BACKENDAI_KERNEL_WORK_DIR`, `BACKENDAI_KERNEL_CONFIG_DIR`, and `BACKENDAI_KERNEL_BIND_HOST` | `execute`, services, and model services work without a second code path in the agent. The runner is Python and already reads its ports from `intrinsic-ports.json` | Run batch commands from the agent without the runner (needs an in-agent replacement for the runner protocol) |
| 2026-10-02 | No isolation; `backend = "native"` is the opt-in; stated in the startup log | macOS has no cgroups or namespaces; `sandbox-exec` is marked deprecated | `sandbox-exec` profile per kernel; a separate OS user per kernel; macOS guest VM per kernel (two-VM license limit) |
| 2026-10-02 | Backend name `native` | BEP-1016 names the workload "(native) process tree"; the backend is not macOS-specific | `process`, `macos`, `host` |
| 2026-10-02 | Service host port equals the declared port | No NAT exists for a host process. Precedent in the Docker host-network branch | Remap every service port through the port pool (the runner and service definitions take declared ports) |
| 2026-10-02 | A mount outside `/home/work` maps to `<scratch-root>/<kernel-id>/mounts/<path>`; the model path of an inference session is rewritten to it | Model vfolders mount at `/models`; refusing such mounts blocked inference sessions | Refuse mounts outside `/home/work`; require `mount_destination` under `/home/work` |
| 2026-10-03 | A runner that dies without a destroy fails the session with `self-terminated`, decided by `exit_handled` in `native-kernel.json` | Without a result event the manager ends the session as if it had finished. The flag on disk also covers a death while the agent is down | Decide by exit code (the agent is not the parent of a re-adopted runner) |
| 2026-10-03 | Running kernels are re-adopted after an agent restart, through the pickle registry file | The records on disk already hold ports and allocations, and the base agent restores from them once the registry loads. A deployment keeps answering across the restart | Terminate leftovers (first cut); rebuild the registry from the records alone |
| 2026-10-03 | Serving runtime variant is `mlx-lm`; `mlxcel` v0.7.0 is not added | rapsealk/backend.ai#21: `mlxcel-server` v0.7.0 returns different greedy output under its default flags and aborts on prompt-cache reuse; it has no training command and no CPU-only Linux build. Revisit when the defect is fixed in a release | `mlxcel` as the variant; both variants |

## Open Questions

| # | Question |
|---|---|
| 1 | **OS dimension in scheduling.** The scheduler matches `architecture` only, so a native macOS agent and a Linux `aarch64` agent in one resource group receive each other's images. Options: a distinct `architecture` value, an OS field on agent and image, or keep the dedicated resource group |
| 2 | **Image definition.** A host path in an image label ties the image to one host layout. Options: agent-side allowlist mapping image → environment, a registry type for host environments, environment provisioning by the agent |
| 3 | **Slot model under unified memory.** Whether `metal.device` deducts from `mem`, or `mem` is capped at physical memory minus the Metal working set |
| 4 | **Migration to `ComputeBackend`.** When BEP-1057 Phase 1 migrates Docker, whether `InstanceSpec.image` stays mandatory for a process tree, and how `InstanceAttachments.mounts` maps to symlinks |
| 5 | **Per-kernel stats.** No cgroup exists. Options: sum over the process tree, or node stats only. Per-process GPU utilization has no public API |
| 6 | **Enforcement.** Whether `mem` is enforced at all (MLX `set_memory_limit` is cooperative; `setrlimit` covers the process, not the GPU working set) |
| 7 | **Registry persistence.** Re-adoption depends on the pickle registry file, which BEP-1002 plans to remove. Whether the records in `native-kernel.json` alone can rebuild the registry |
| 8 | **Linux hosts.** Whether `native` is supported on Linux, where containers already reach the GPU |

## References

- [BEP-1002: Agent Architecture](BEP-1002-agent-architecture.md)
- [BEP-1016: Accelerator Interface v2](BEP-1016-accelerator-interface-v2.md)
- [BEP-1051: Kata Containers Agent](BEP-1051-kata-containers-agent.md)
- [BEP-1057: Agent Re-architecture](BEP-1057-agent-re-architecture.md)
- [MLX build and install](https://ml-explore.github.io/mlx/build/html/install.html) — `mlx[cpu]`, `mlx[cuda12]` on Linux
- [mlx-lm `LORA.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md), [`SERVER.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- [Docker Model Runner sandbox profile](https://github.com/docker/model-runner/blob/main/pkg/sandbox/sandbox_darwin.go)
- [`MTLDevice.recommendedMaxWorkingSetSize`](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
