from __future__ import annotations

import plistlib
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai.backend.accelerator.metal import plugin as plugin_module
from ai.backend.accelerator.metal.ioreg import parse_accelerators
from ai.backend.accelerator.metal.plugin import MetalPlugin
from ai.backend.agent.types import Container
from ai.backend.common.types import (
    ContainerId,
    ContainerStatus,
    DeviceId,
    MetricKey,
    SlotName,
)

# `ioreg -a -r -d 1 -w 0 -c IOAccelerator` on an Apple M5 Pro (macOS 27.0),
# trimmed to the keys the plugin reads plus a few of the unrelated ones.
RECORDED_IOREG = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
	<dict>
		<key>AGXParameterBufferMaxSize</key>
		<integer>-1717043200</integer>
		<key>IOClass</key>
		<string>AGXAcceleratorG17X</string>
		<key>IOGeneralInterest</key>
		<string>IOCommand is not serializable</string>
		<key>IONameMatched</key>
		<string>gpu,t6050</string>
		<key>IORegistryEntryName</key>
		<string>AGXAcceleratorG17X</string>
		<key>PerformanceStatistics</key>
		<dict>
			<key>Alloc system memory</key>
			<integer>6614056960</integer>
			<key>Allocated PB Size</key>
			<integer>214958080</integer>
			<key>Device Utilization %</key>
			<integer>12</integer>
			<key>In use system memory</key>
			<integer>1046872064</integer>
			<key>In use system memory (driver)</key>
			<integer>0</integer>
			<key>Renderer Utilization %</key>
			<integer>12</integer>
			<key>SplitSceneCount</key>
			<integer>0</integer>
			<key>TiledSceneBytes</key>
			<integer>2129920</integer>
			<key>Tiler Utilization %</key>
			<integer>5</integer>
			<key>lastRecoveryTime</key>
			<integer>0</integer>
			<key>recoveryCount</key>
			<integer>0</integer>
		</dict>
		<key>gpu-core-count</key>
		<integer>16</integer>
		<key>model</key>
		<string>Apple M5 Pro</string>
		<key>vendor-id</key>
		<data>
		axAAAA==
		</data>
	</dict>
</array>
</plist>
"""
RECOMMENDED_WORKING_SET_SIZE = 40200896512
PHYSICAL_MEMORY_SIZE = 51539607552
SLOT = SlotName("metal.device")
MEM = MetricKey("metal_mem")
UTIL = MetricKey("metal_util")


def _patch_host(
    monkeypatch: pytest.MonkeyPatch,
    *,
    apple_silicon: bool = True,
    accelerators: Sequence[Mapping[str, Any]] | None = None,
    working_set_size: int | None = RECOMMENDED_WORKING_SET_SIZE,
) -> None:
    entries = parse_accelerators(RECORDED_IOREG) if accelerators is None else accelerators

    async def read_accelerators() -> Sequence[Mapping[str, Any]]:
        return entries

    monkeypatch.setattr(plugin_module, "is_apple_silicon", lambda: apple_silicon)
    monkeypatch.setattr(plugin_module, "read_accelerators", read_accelerators)
    monkeypatch.setattr(plugin_module, "recommended_working_set_size", lambda: working_set_size)
    monkeypatch.setattr(plugin_module, "physical_memory_size", lambda: PHYSICAL_MEMORY_SIZE)


async def _init_plugin() -> MetalPlugin:
    plugin = MetalPlugin({}, {})
    await plugin.init()
    return plugin


class TestParseAccelerators:
    def test_recorded_output(self) -> None:
        (entry,) = parse_accelerators(RECORDED_IOREG)

        assert entry["model"] == "Apple M5 Pro"
        assert entry["gpu-core-count"] == 16
        assert entry["PerformanceStatistics"]["Device Utilization %"] == 12

    @pytest.mark.parametrize(
        "raw",
        [b"", b"\n", b"not a plist", b"<plist><array><dict>", plistlib.dumps({"model": "x"})],
        ids=["empty", "blank", "garbage", "truncated", "not-a-list"],
    )
    def test_unusable_output_yields_no_entries(self, raw: bytes) -> None:
        assert parse_accelerators(raw) == []


class TestMetalPluginOnAppleSilicon:
    async def test_reports_one_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_host(monkeypatch)

        plugin = await _init_plugin()

        (device,) = await plugin.list_devices()
        assert device.device_name == "metal"
        assert device.device_id == DeviceId("0")
        assert device.model_name == "Apple M5 Pro"
        assert device.hw_location == "gpu,t6050"
        assert device.processing_units == 16
        assert device.memory_size == RECOMMENDED_WORKING_SET_SIZE
        assert await plugin.available_slots() == {SLOT: Decimal(1)}
        assert (await plugin.extra_info())["metal_support"] == "true"

    async def test_memory_falls_back_to_physical_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_host(monkeypatch, working_set_size=None)

        plugin = await _init_plugin()

        (device,) = await plugin.list_devices()
        assert device.memory_size == PHYSICAL_MEMORY_SIZE

    async def test_allocates_the_device_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_host(monkeypatch)
        plugin = await _init_plugin()
        alloc_map = await plugin.create_alloc_map()

        device_alloc = alloc_map.allocate({SLOT: Decimal(1)})

        assert device_alloc == {SLOT: {DeviceId("0"): Decimal(1)}}
        (attached,) = await plugin.get_attached_devices(device_alloc)
        assert attached["model_name"] == "Apple M5 Pro"
        assert attached["data"] == {"proc": 16, "mem": RECOMMENDED_WORKING_SET_SIZE}
        assert await plugin.get_attached_devices({}) == []

    async def test_node_measures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_host(monkeypatch)
        plugin = await _init_plugin()

        measures = {m.key: m for m in await plugin.gather_node_measures(MagicMock())}

        assert measures.keys() == {MEM, UTIL}
        assert measures[MEM].per_node.value == Decimal(1046872064)
        assert measures[MEM].per_node.capacity == Decimal(RECOMMENDED_WORKING_SET_SIZE)
        assert measures[UTIL].per_node.value == Decimal(12)
        assert measures[UTIL].per_device.keys() == {DeviceId("0")}

    @pytest.mark.parametrize(
        ("statistics", "expected_keys"),
        [
            (None, set()),
            ({}, set()),
            ({"Device Utilization %": 3}, {"metal_util"}),
            ({"In use system memory": 1024}, {"metal_mem"}),
        ],
        ids=["no-statistics", "empty", "utilization-only", "memory-only"],
    )
    async def test_node_measures_tolerate_missing_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
        statistics: Mapping[str, int] | None,
        expected_keys: set[str],
    ) -> None:
        entry: dict[str, Any] = {"model": "Apple M5 Pro"}
        if statistics is not None:
            entry["PerformanceStatistics"] = statistics
        _patch_host(monkeypatch, accelerators=[entry])
        plugin = await _init_plugin()

        measures = await plugin.gather_node_measures(MagicMock())

        assert {m.key for m in measures} == expected_keys

    async def test_restore_ignores_a_non_docker_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_host(monkeypatch)
        plugin = await _init_plugin()
        alloc_map = await plugin.create_alloc_map()
        container = Container(
            id=ContainerId("native-kernel"),
            status=ContainerStatus.RUNNING,
            image="",
            labels={},
            ports=[],
            backend_obj=None,
        )

        await plugin.restore_from_container(container, alloc_map)

        assert alloc_map.allocations[SLOT][DeviceId("0")] == Decimal(0)


class TestMetalPluginElsewhere:
    @pytest.mark.parametrize(
        ("apple_silicon", "accelerators"),
        [(False, None), (True, []), (True, [{"IOClass": "no-model"}])],
        ids=["other-platform", "no-accelerator", "no-model"],
    )
    async def test_is_inert(
        self,
        monkeypatch: pytest.MonkeyPatch,
        apple_silicon: bool,
        accelerators: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        _patch_host(monkeypatch, apple_silicon=apple_silicon, accelerators=accelerators)

        plugin = await _init_plugin()

        assert list(await plugin.list_devices()) == []
        assert await plugin.available_slots() == {SLOT: Decimal(0)}
        assert await plugin.gather_node_measures(MagicMock()) == []
        assert (await plugin.create_alloc_map()).device_slots == {}
        assert await plugin.extra_info() == {"metal_support": "false"}
