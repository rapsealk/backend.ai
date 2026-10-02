from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import os
import platform
import plistlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any
from xml.parsers.expat import ExpatError

_METAL_FRAMEWORK = "/System/Library/Frameworks/Metal.framework/Metal"


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def parse_accelerators(raw: bytes) -> Sequence[Mapping[str, Any]]:
    """Parse `ioreg -a` output into IOAccelerator entries; malformed input yields none."""
    if not raw.strip():
        return []
    try:
        entries = plistlib.loads(raw)
    except (plistlib.InvalidFileException, ExpatError, ValueError):
        return []
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


async def read_accelerators() -> Sequence[Mapping[str, Any]]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *("ioreg", "-a", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return []
    stdout, _ = await proc.communicate()
    return parse_accelerators(stdout)


def physical_memory_size() -> int:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def recommended_working_set_size() -> int | None:
    """`MTLDevice.recommendedMaxWorkingSetSize` of the default Metal device, or None."""
    if not is_apple_silicon():
        return None
    try:
        metal = ctypes.CDLL(_METAL_FRAMEWORK)
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        msg_send = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)((
            "objc_msgSend",
            objc,
        ))
        device = metal.MTLCreateSystemDefaultDevice()
        if not device:
            return None
        size = msg_send(device, objc.sel_registerName(b"recommendedMaxWorkingSetSize"))
    except (OSError, AttributeError):
        return None
    return size or None
