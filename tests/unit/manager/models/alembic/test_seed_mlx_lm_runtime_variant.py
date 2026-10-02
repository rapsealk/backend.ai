"""Runs the mlx-lm runtime variant seed against a real database and reads the rows back."""

from __future__ import annotations

import json
import pathlib
from collections.abc import AsyncGenerator

import pytest
import sqlalchemy as sa
from sqlalchemy import Table

from ai.backend.common.data.entity.global_entity import GlobalEntityName
from ai.backend.common.data.entity.runtime_variant import RuntimeVariantEntityType
from ai.backend.common.data.entity.runtime_variant_preset import RuntimeVariantPresetEntityType
from ai.backend.common.data.entity.virtual_entity import VirtualEntityID
from ai.backend.manager.data.permission.global_entity import global_entity_id
from ai.backend.manager.models.alembic.versions import (
    c686add49059_seed_mlx_lm_runtime_variant as revision,
)
from ai.backend.manager.models.runtime_variant.row import RuntimeVariantRow
from ai.backend.manager.models.runtime_variant_preset.row import RuntimeVariantPresetRow
from ai.backend.manager.models.utils import ExtendedAsyncSAEngine
from ai.backend.manager.models.virtual_entity.entity_membership import EntityMembershipRow
from ai.backend.manager.models.virtual_entity.scope_binding import ScopeBindingRow
from ai.backend.manager.models.virtual_entity.virtual_entity import VirtualEntityRow
from ai.backend.testutils.db import HasTable, with_tables

_TABLES: list[Table | type[HasTable]] = [RuntimeVariantRow, RuntimeVariantPresetRow]
_FIXTURE_DIR = pathlib.Path("fixtures/manager")


@pytest.fixture
async def db(
    global_entity_ids: ExtendedAsyncSAEngine,
) -> AsyncGenerator[ExtendedAsyncSAEngine, None]:
    async with with_tables(global_entity_ids, _TABLES):
        # A stand-in for the one column the downgrade reads; the real table drags in endpoints.
        async with global_entity_ids.begin() as conn:
            await conn.execute(
                sa.text("CREATE TABLE deployment_revisions (runtime_variant_id uuid NOT NULL)")
            )
        try:
            yield global_entity_ids
        finally:
            async with global_entity_ids.begin() as conn:
                await conn.execute(sa.text("DROP TABLE deployment_revisions"))


async def _upgrade(db: ExtendedAsyncSAEngine) -> None:
    async with db.begin() as conn:
        await conn.run_sync(revision.seed)


async def _downgrade(db: ExtendedAsyncSAEngine) -> None:
    async with db.begin() as conn:
        await conn.run_sync(revision.remove_seed)


async def _variant(db: ExtendedAsyncSAEngine) -> RuntimeVariantRow | None:
    async with db.begin_readonly_session() as session:
        return (
            await session.scalars(
                sa.select(RuntimeVariantRow).where(RuntimeVariantRow.name == revision.VARIANT_NAME)
            )
        ).one_or_none()


async def _presets(db: ExtendedAsyncSAEngine) -> list[RuntimeVariantPresetRow]:
    async with db.begin_readonly_session() as session:
        return list(
            await session.scalars(
                sa.select(RuntimeVariantPresetRow).order_by(RuntimeVariantPresetRow.rank)
            )
        )


async def _scopes_of(db: ExtendedAsyncSAEngine, node: VirtualEntityID) -> set[VirtualEntityID]:
    """The scopes that own and govern the node, read from both edge tables."""
    async with db.begin_readonly_session() as session:
        owners = set(
            await session.scalars(
                sa.select(EntityMembershipRow.virtual_entity_id).where(
                    EntityMembershipRow.member_entity_id == node
                )
            )
        )
        governors = set(
            await session.scalars(
                sa.select(ScopeBindingRow.scope_entity_id).where(
                    ScopeBindingRow.virtual_entity_id == node
                )
            )
        )
    assert owners == governors
    return owners


async def _global_nodes(db: ExtendedAsyncSAEngine) -> set[VirtualEntityID]:
    """The nodes of the `global` and `public` scopes."""
    async with db.begin_readonly_session() as session:
        return set(
            await session.scalars(
                sa.select(VirtualEntityRow.id).where(
                    VirtualEntityRow.entity_type == "global",
                    VirtualEntityRow.entity_id.in_([
                        global_entity_id(GlobalEntityName.GLOBAL),
                        global_entity_id(GlobalEntityName.PUBLIC),
                    ]),
                )
            )
        )


async def _count_nodes(db: ExtendedAsyncSAEngine) -> int:
    async with db.begin_readonly_session() as session:
        return (await session.scalar(sa.select(sa.func.count()).select_from(VirtualEntityRow))) or 0


class TestSeedMlxLmRuntimeVariant:
    async def test_the_variant_serves_with_mlx_lm_server(self, db: ExtendedAsyncSAEngine) -> None:
        await _upgrade(db)

        variant = await _variant(db)
        assert variant is not None
        models = variant.default_model_definition.models
        assert models is not None
        service = models[0].service
        assert service.start_command == (
            "python -m mlx_lm server --model '{model_path}' --host 0.0.0.0 --port 8080"
        )
        assert service.port == 8080
        assert service.health_check is not None
        assert service.health_check.path == "/health"
        assert service.pre_start_actions is not None
        assert [(action.action, action.args) for action in service.pre_start_actions] == [
            ("mkdir", {"path": ".cache/huggingface/hub"})
        ]

    async def test_every_preset_is_written_for_the_variant(self, db: ExtendedAsyncSAEngine) -> None:
        await _upgrade(db)

        variant = await _variant(db)
        assert variant is not None
        presets = await _presets(db)
        assert [preset.name for preset in presets] == [p["name"] for p in revision.PRESETS]
        assert {preset.runtime_variant for preset in presets} == {variant.id}

    async def test_each_row_is_owned_by_itself_global_and_public(
        self, db: ExtendedAsyncSAEngine
    ) -> None:
        await _upgrade(db)

        variant = await _variant(db)
        assert variant is not None
        entities = [(RuntimeVariantEntityType(), variant.id)] + [
            (RuntimeVariantPresetEntityType(), preset.id) for preset in await _presets(db)
        ]
        global_nodes = await _global_nodes(db)
        assert len(global_nodes) == 2
        for entity_type, entity_id in entities:
            async with db.begin_readonly_session() as session:
                node = await session.scalar(
                    sa.select(VirtualEntityRow.id).where(
                        VirtualEntityRow.entity_type == entity_type,
                        VirtualEntityRow.entity_id == entity_id,
                    )
                )
            assert node is not None
            assert await _scopes_of(db, node) == {node, *global_nodes}

    async def test_running_it_twice_leaves_the_same_rows(self, db: ExtendedAsyncSAEngine) -> None:
        await _upgrade(db)
        once = (await _count_nodes(db), len(await _presets(db)))

        await _upgrade(db)

        assert (await _count_nodes(db), len(await _presets(db))) == once

    async def test_rows_an_operator_made_are_left_alone(self, db: ExtendedAsyncSAEngine) -> None:
        async with db.begin() as conn:
            await conn.execute(
                sa.text("""
                    INSERT INTO runtime_variants (name, description, default_model_definition)
                    VALUES (
                        :name, 'operator',
                        '{"models": [{"name": "operator", "service": {"port": 9999}}]}'
                    )
                """),
                {"name": revision.VARIANT_NAME},
            )
            await conn.execute(
                sa.text("""
                    INSERT INTO runtime_variant_presets
                        (runtime_variant, name, rank, preset_target, value_type, default_value, key)
                    SELECT id, 'adapter-path', 1, 'args', 'str', '/operator', '--adapter-path'
                    FROM runtime_variants WHERE name = :name
                """),
                {"name": revision.VARIANT_NAME},
            )

        await _upgrade(db)

        variant = await _variant(db)
        assert variant is not None
        assert variant.description == "operator"
        models = variant.default_model_definition.models
        assert models is not None
        assert models[0].service.port == 9999
        presets = {preset.name: preset for preset in await _presets(db)}
        assert presets["adapter-path"].default_value == "/operator"
        assert set(presets) == {p["name"] for p in revision.PRESETS}

    async def test_the_downgrade_takes_back_what_it_wrote(self, db: ExtendedAsyncSAEngine) -> None:
        before = await _count_nodes(db)
        await _upgrade(db)

        await _downgrade(db)

        assert await _variant(db) is None
        assert await _presets(db) == []
        assert await _count_nodes(db) == before

    async def test_the_downgrade_keeps_a_variant_a_revision_names(
        self, db: ExtendedAsyncSAEngine
    ) -> None:
        await _upgrade(db)
        variant = await _variant(db)
        assert variant is not None
        async with db.begin() as conn:
            await conn.execute(
                sa.text("INSERT INTO deployment_revisions (runtime_variant_id) VALUES (:id)"),
                {"id": variant.id},
            )

        await _downgrade(db)

        assert await _variant(db) is not None
        assert await _presets(db) == []


class TestFixtureMatchesTheSeed:
    """An install loads the fixture and an upgrade runs the seed; both must write the same rows."""

    def test_the_variant_row(self) -> None:
        rows = json.loads((_FIXTURE_DIR / "example-runtime-variants.json").read_text())
        variant = next(
            row for row in rows["runtime_variants"] if row["name"] == revision.VARIANT_NAME
        )
        assert variant["default_model_definition"] == revision.DEFAULT_MODEL_DEFINITION

    def test_the_preset_rows(self) -> None:
        rows = json.loads((_FIXTURE_DIR / "example-runtime-variant-presets.json").read_text())
        presets = [
            {key: value for key, value in row.items() if key != "runtime_variant_name"}
            for row in rows["runtime_variant_presets"]
            if row["runtime_variant_name"] == revision.VARIANT_NAME
        ]
        assert presets == revision.PRESETS
