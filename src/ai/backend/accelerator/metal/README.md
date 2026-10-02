# backend.ai-accelerator-metal

Reports the Apple Silicon GPU to the Backend.AI agent as a schedulable device.

| Item | Value |
|---|---|
| Plugin key | `metal` |
| Slot | `metal.device` (count), one device per host |
| Device memory | Metal `recommendedMaxWorkingSetSize`; physical memory size when Metal is unavailable |
| Node metrics | `metal_util` (percent), `metal_mem` (bytes) from `ioreg -c IOAccelerator` |
| Other platforms | No device, no slot, no error |

## Agent configuration

```toml
[agent]
allow-compute-plugins = ["ai.backend.accelerator.metal"]

[resource]
allocation-order = ["metal", "cpu", "mem"]
```

`allocation-order` must list `metal`; kernel creation fails for a slot whose device name is missing from it.

## Limits

- A Linux container cannot use Metal. Under the Docker backend the plugin only advertises and accounts the slot.
- The GPU shares the host's unified memory, so the device memory overlaps the `mem` slot.
- The allocation is not enforced; any host process can use the GPU.
- macOS exposes no per-process GPU accounting, so there are no per-kernel GPU metrics.
