"""Images the manager has handed to this agent, kept so that scan_images() can report them."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel

from ai.backend.agent.agent import ScanImagesResult
from ai.backend.common.data.image.types import InstalledImageInfo
from ai.backend.common.docker import LabelName
from ai.backend.common.types import ImageCanonical, ImageConfig

IMAGES_FILENAME = "native-images.{agent_id}.json"


class ImageRecord(BaseModel):
    architecture: str
    digest: str
    runtime_path: str

    def is_installed(self) -> bool:
        return Path(self.runtime_path).is_absolute() and os.access(self.runtime_path, os.X_OK)


class ImageRecords(BaseModel):
    images: dict[str, ImageRecord] = {}


def images_file(var_base_path: Path, agent_id: str) -> Path:
    return var_base_path / IMAGES_FILENAME.format(agent_id=agent_id)


def read_records(path: Path) -> ImageRecords:
    try:
        return ImageRecords.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        return ImageRecords()


def record_images(path: Path, configs: Iterable[ImageConfig]) -> None:
    records = read_records(path)
    for conf in configs:
        records.images[conf["canonical"]] = ImageRecord(
            architecture=conf["architecture"],
            digest=conf["digest"],
            runtime_path=conf["labels"].get(LabelName.RUNTIME_PATH, ""),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(records.model_dump_json())
    tmp_path.replace(path)


def scan_records(
    path: Path, reported: Mapping[ImageCanonical, InstalledImageInfo]
) -> ScanImagesResult:
    """An image is installed while its runtime path is an executable on this host."""
    scanned = {
        ImageCanonical(canonical): InstalledImageInfo(
            canonical=canonical, digest=rec.digest, architecture=rec.architecture
        )
        for canonical, rec in read_records(path).images.items()
        if rec.is_installed()
    }
    removed = {c: info for c, info in reported.items() if c not in scanned}
    return ScanImagesResult(scanned_images=scanned, removed_images=removed)
