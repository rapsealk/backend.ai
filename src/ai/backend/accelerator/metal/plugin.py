from __future__ import annotations

import logging
import platform
from collections.abc import Collection, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, override

import aiodocker

from ai.backend.agent.docker.resources import get_resource_spec_from_container
from ai.backend.agent.resources import (
    AbstractAllocMap,
    AbstractComputeDevice,
    AbstractComputePlugin,
    DeviceSlotInfo,
    DiscretePropertyAllocMap,
)
from ai.backend.agent.stats import (
    ContainerMeasurement,
    Measurement,
    MetricTypes,
    NodeMeasurement,
    ProcessMeasurement,
    StatContext,
)
from ai.backend.agent.types import Container, MountInfo
from ai.backend.common.types import (
    AcceleratorMetadata,
    BinarySize,
    DeviceId,
    DeviceModelInfo,
    DeviceName,
    MetricKey,
    SlotName,
    SlotTypes,
)
from ai.backend.logging.structured import StructuredLogger

from . import __version__
from .ioreg import (
    is_apple_silicon,
    physical_memory_size,
    read_accelerators,
    recommended_working_set_size,
)

log = StructuredLogger(logging.getLogger(__spec__.name))

SLOT_NAME = SlotName("metal.device")


class MetalDevice(AbstractComputeDevice):
    model_name: str

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name


class MetalPlugin(AbstractComputePlugin):
    key = DeviceName("metal")
    slot_types: Sequence[tuple[SlotName, SlotTypes]] = ((SLOT_NAME, SlotTypes.COUNT),)
    exclusive_slot_types: set[str] = {SLOT_NAME}

    enabled: bool = False
    _devices: Sequence[MetalDevice] = ()

    @override
    async def init(self, context: Any = None) -> None:
        if not is_apple_silicon():
            log.info("accelerator disabled: not an Apple Silicon host", plugin_name=self.key)
            return
        accelerators = [entry for entry in await read_accelerators() if "model" in entry]
        if not accelerators:
            log.warning("accelerator disabled: no IOAccelerator device found", plugin_name=self.key)
            return
        # Apple Silicon has one GPU; it shares the host's unified memory.
        entry = accelerators[0]
        self._devices = [
            MetalDevice(
                model_name=str(entry["model"]),
                device_id=DeviceId("0"),
                hw_location=str(entry.get("IONameMatched", "gpu")),
                memory_size=recommended_working_set_size() or physical_memory_size(),
                processing_units=int(entry.get("gpu-core-count", 1)),
            )
        ]
        self.enabled = True

    @override
    async def cleanup(self) -> None:
        pass

    @override
    async def update_plugin_config(self, plugin_config: Mapping[str, Any]) -> None:
        pass

    @override
    def get_metadata(self) -> AcceleratorMetadata:
        return {
            "slot_name": str(SLOT_NAME),
            "human_readable_name": "GPU",
            "description": "Apple GPU (Metal)",
            "display_unit": "GPU",
            "number_format": {"binary": False, "round_length": 0},
            "display_icon": "gpu1",
        }

    @override
    async def list_devices(self) -> Collection[MetalDevice]:
        return self._devices

    @override
    async def available_slots(self) -> Mapping[SlotName, Decimal]:
        return {SLOT_NAME: Decimal(len(self._devices))}

    @override
    def get_version(self) -> str:
        return __version__

    @override
    async def extra_info(self) -> Mapping[str, str]:
        if not self.enabled:
            return {"metal_support": "false"}
        return {"metal_support": "true", "macos_version": platform.mac_ver()[0]}

    @override
    async def gather_node_measures(self, ctx: StatContext) -> Sequence[NodeMeasurement]:
        if not self._devices:
            return []
        device = self._devices[0]
        # The PerformanceStatistics keys are undocumented; report only those present.
        stats: Mapping[str, Any] = {}
        for entry in await read_accelerators():
            if isinstance(found := entry.get("PerformanceStatistics"), dict):
                stats = found
                break
        measures: list[NodeMeasurement] = []
        if (mem_used := stats.get("In use system memory")) is not None:
            mem = Measurement(Decimal(mem_used), Decimal(device.memory_size))
            measures.append(
                NodeMeasurement(
                    MetricKey("metal_mem"),
                    MetricTypes.GAUGE,
                    unit_hint="bytes",
                    stats_filter=frozenset({"max"}),
                    per_node=mem,
                    per_device={device.device_id: mem},
                )
            )
        if (util := stats.get("Device Utilization %")) is not None:
            usage = Measurement(Decimal(util), Decimal(100))
            measures.append(
                NodeMeasurement(
                    MetricKey("metal_util"),
                    MetricTypes.GAUGE,
                    unit_hint="percent",
                    stats_filter=frozenset({"avg", "max"}),
                    per_node=usage,
                    per_device={device.device_id: usage},
                )
            )
        return measures

    @override
    async def gather_container_measures(
        self,
        ctx: StatContext,
        container_ids: Sequence[str],
    ) -> Sequence[ContainerMeasurement]:
        # macOS exposes no per-process GPU accounting.
        return []

    @override
    async def gather_process_measures(
        self, ctx: StatContext, pid_map: Mapping[int, str]
    ) -> Sequence[ProcessMeasurement]:
        return []

    @override
    async def create_alloc_map(self) -> AbstractAllocMap:
        return DiscretePropertyAllocMap(
            device_slots={
                dev.device_id: DeviceSlotInfo(SlotTypes.COUNT, SLOT_NAME, Decimal(1))
                for dev in self._devices
            },
        )

    @override
    async def get_hooks(self, distro: str, arch: str) -> Sequence[Path]:
        return []

    @override
    async def generate_docker_args(
        self,
        docker: aiodocker.docker.Docker,
        device_alloc: Mapping[SlotName, Mapping[DeviceId, Decimal]],
    ) -> Mapping[str, Any]:
        # A Linux container cannot reach Metal; there is nothing to attach.
        return {}

    @override
    async def restore_from_container(
        self,
        container: Container,
        alloc_map: AbstractAllocMap,
    ) -> None:
        # Only a Docker-shaped backend object carries the mounted resource.txt.
        if not self.enabled or not isinstance(container.backend_obj, Mapping):
            return
        resource_spec = await get_resource_spec_from_container(container.backend_obj)
        if resource_spec is None:
            return
        if (alloc := resource_spec.allocations.get(self.key, {}).get(SLOT_NAME)) is not None:
            alloc_map.apply_allocation({SLOT_NAME: alloc})

    @override
    async def get_attached_devices(
        self,
        device_alloc: Mapping[SlotName, Mapping[DeviceId, Decimal]],
    ) -> Sequence[DeviceModelInfo]:
        device_ids = device_alloc.get(SLOT_NAME, {}).keys()
        return [
            {
                "device_id": device.device_id,
                "model_name": device.model_name,
                "data": {
                    "proc": device.processing_units,
                    "mem": BinarySize(device.memory_size),
                },
            }
            for device in self._devices
            if device.device_id in device_ids
        ]

    @override
    async def get_docker_networks(
        self, device_alloc: Mapping[SlotName, Mapping[DeviceId, Decimal]]
    ) -> list[str]:
        return []

    @override
    async def generate_mounts(
        self, source_path: Path, device_alloc: Mapping[SlotName, Mapping[DeviceId, Decimal]]
    ) -> list[MountInfo]:
        return []
