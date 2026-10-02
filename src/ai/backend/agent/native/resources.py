from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from ai.backend.agent.docker.resources import scan_available_resources
from ai.backend.agent.errors import InitializationError
from ai.backend.agent.resources import AbstractComputePlugin, ComputePluginContext
from ai.backend.common.etcd import AbstractKVStore
from ai.backend.common.types import DeviceName

from .intrinsic import CPUPlugin, MemoryPlugin

__all__ = ("load_resources", "scan_available_resources")


async def load_resources(
    etcd: AbstractKVStore,
    local_config: Mapping[str, Any],
) -> Mapping[DeviceName, AbstractComputePlugin]:
    compute_device_types: MutableMapping[DeviceName, AbstractComputePlugin] = {}
    compute_plugin_ctx = ComputePluginContext(etcd, local_config)
    await compute_plugin_ctx.init(
        allowlist=local_config["agent"]["allow-compute-plugins"],
        blocklist=local_config["agent"]["block-compute-plugins"],
    )
    if "cpu" not in compute_plugin_ctx.plugins:
        cpu_plugin = CPUPlugin(await etcd.get_prefix("config/plugins/cpu"), local_config)
        await cpu_plugin.init()
        compute_plugin_ctx.attach_intrinsic_device(cpu_plugin)
    if "mem" not in compute_plugin_ctx.plugins:
        memory_plugin = MemoryPlugin(await etcd.get_prefix("config/plugins/memory"), local_config)
        await memory_plugin.init()
        compute_plugin_ctx.attach_intrinsic_device(memory_plugin)
    for plugin_instance in compute_plugin_ctx.plugins.values():
        for slot_name, _ in plugin_instance.slot_types:
            if slot_name not in {"cpu", "mem"} and not slot_name.startswith(
                f"{plugin_instance.key}."
            ):
                raise InitializationError(
                    "Slot types defined by an accelerator plugin must be prefixed by the plugin's key. "
                    f"(invalid slot: {slot_name!r}, plugin key: {plugin_instance.key!r})"
                )
        if plugin_instance.key in compute_device_types:
            raise InitializationError(
                f"A plugin defining the same key '{plugin_instance.key}' already exists. "
                "You may need to uninstall it first."
            )
        compute_device_types[plugin_instance.key] = plugin_instance
    return compute_device_types
