"""CPU and memory plugins for the native backend: the Docker ones without Docker or cgroups."""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping, Sequence
from typing import Any, override

from ai.backend.agent.docker.intrinsic import CPUDevice
from ai.backend.agent.docker.intrinsic import CPUPlugin as DockerCPUPlugin
from ai.backend.agent.docker.intrinsic import MemoryPlugin as DockerMemoryPlugin
from ai.backend.agent.docker.resources import get_resource_spec_from_container
from ai.backend.agent.resources import AbstractAllocMap
from ai.backend.agent.stats import ContainerMeasurement, ProcessMeasurement, StatContext
from ai.backend.agent.types import Container
from ai.backend.agent.vendor.linux import get_cpus, libnuma
from ai.backend.common.types import DeviceId, DeviceName, SlotName


class CPUPlugin(DockerCPUPlugin):
    @override
    async def init(self, context: Any | None = None) -> None:
        pass

    @override
    async def cleanup(self) -> None:
        pass

    @override
    async def list_devices(self) -> Collection[CPUDevice]:
        # The cores the agent itself may use; there is no Docker VM or cgroup in between.
        cores = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else get_cpus()
        return [
            CPUDevice(
                device_id=DeviceId(str(core_idx)),
                hw_location="root",
                numa_node=libnuma.node_of_cpu(core_idx),
                memory_size=0,
                processing_units=1,
            )
            for core_idx in sorted(cores)
        ]

    @override
    async def gather_container_measures(
        self, ctx: StatContext, container_ids: Sequence[str]
    ) -> Sequence[ContainerMeasurement]:
        return []

    @override
    async def gather_process_measures(
        self, ctx: StatContext, pid_map: Mapping[int, str]
    ) -> Sequence[ProcessMeasurement]:
        return []


class MemoryPlugin(DockerMemoryPlugin):
    @override
    async def init(self, context: Any | None = None) -> None:
        self._graph_root_prefix = None

    @override
    async def cleanup(self) -> None:
        pass

    @override
    async def _get_graph_root_prefix(self) -> str | None:
        return None

    @override
    async def gather_container_measures(
        self, ctx: StatContext, container_ids: Sequence[str]
    ) -> Sequence[ContainerMeasurement]:
        return []

    @override
    async def gather_process_measures(
        self, ctx: StatContext, pid_map: Mapping[int, str]
    ) -> Sequence[ProcessMeasurement]:
        return []

    @override
    async def restore_from_container(
        self,
        container: Container,
        alloc_map: AbstractAllocMap,
    ) -> None:
        # No container memory limit to read back; use our own record as the CPU plugin does.
        resource_spec = await get_resource_spec_from_container(container.backend_obj)
        if resource_spec is None:
            return
        alloc_map.apply_allocation({
            SlotName("mem"): resource_spec.allocations[DeviceName("mem")][SlotName("mem")],
        })
