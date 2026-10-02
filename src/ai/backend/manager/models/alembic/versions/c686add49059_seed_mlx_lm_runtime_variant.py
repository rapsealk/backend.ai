"""seed the mlx-lm runtime variant

Adds the ``mlx-lm`` runtime variant, which serves a model folder with
``mlx_lm server``, and its presets. Each row gets the graph node the entity
ops would have provisioned. Rows an operator already created are left as they are.

Revision ID: c686add49059
Revises: d17b4e9c25a8
Create Date: 2026-10-02

"""

import json
from typing import Any, Final

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c686add49059"  # Part of: NEXT_RELEASE_VERSION
down_revision = "d17b4e9c25a8"
branch_labels = None
depends_on = None

VARIANT_NAME: Final[str] = "mlx-lm"

DEFAULT_MODEL_DEFINITION: Final[dict[str, Any]] = {
    "models": [
        {
            "name": "mlx-lm-model",
            "service": {
                # GET /v1/models returns an empty 200 without the hub cache directory
                # (ml-explore/mlx-lm#1636); the path is relative to the kernel home.
                "pre_start_actions": [
                    {"action": "mkdir", "args": {"path": ".cache/huggingface/hub"}},
                ],
                "start_command": (
                    "python -m mlx_lm server --model '{model_path}' --host 0.0.0.0 --port 8080"
                ),
                "port": 8080,
                "health_check": {
                    "enable": True,
                    "path": "/health",
                    "interval": 10.0,
                    "max_retries": 10,
                    "initial_delay": 1800.0,
                },
            },
        }
    ]
}

# Snapshot of the mlx-lm rows of fixtures/manager/example-runtime-variant-presets.json.
PRESETS: Final[list[dict[str, Any]]] = [
    {
        "name": "adapter-path",
        "description": "Path to trained adapter (LoRA) weights applied on top of the model.",
        "rank": 100,
        "preset_target": "args",
        "value_type": "str",
        "default_value": None,
        "key": "--adapter-path",
        "category": "lora",
        "display_name": "Adapter Path",
        "ui_option": {"ui_type": "text_input", "text": {"placeholder": "/models/adapters"}},
    },
    {
        "name": "trust-remote-code",
        "description": "Trust remote code when loading the tokenizer.",
        "rank": 200,
        "preset_target": "args",
        "value_type": "flag",
        "default_value": None,
        "key": "--trust-remote-code",
        "category": "model_loading",
        "display_name": "Trust Remote Code",
        "ui_option": {"ui_type": "checkbox"},
    },
    {
        "name": "max-tokens",
        "description": "Default maximum number of tokens to generate per request.",
        "rank": 300,
        "preset_target": "args",
        "value_type": "int",
        "default_value": "512",
        "key": "--max-tokens",
        "category": "serving_performance",
        "display_name": "Max Tokens",
        "ui_option": {"ui_type": "number_input", "number": {"min": 1, "max": 131072}},
    },
    {
        "name": "chat-template-args",
        "description": "JSON arguments passed to the tokenizer's chat template.",
        "rank": 400,
        "preset_target": "args",
        "value_type": "str",
        "default_value": None,
        "key": "--chat-template-args",
        "category": "tool_reasoning",
        "display_name": "Chat Template Args",
        "ui_option": {
            "ui_type": "text_input",
            "text": {"placeholder": '{"enable_thinking": false}'},
        },
    },
    {
        "name": "log-level",
        "description": "Logging level of the server.",
        "rank": 500,
        "preset_target": "args",
        "value_type": "str",
        "default_value": "INFO",
        "key": "--log-level",
        "category": "monitoring",
        "display_name": "Log Level",
        "ui_option": {
            "ui_type": "select",
            "choices": {
                "items": [
                    {"value": "DEBUG", "label": "Debug"},
                    {"value": "INFO", "label": "Info"},
                    {"value": "WARNING", "label": "Warning"},
                    {"value": "ERROR", "label": "Error"},
                    {"value": "CRITICAL", "label": "Critical"},
                ]
            },
        },
    },
]

_INSERT_VARIANT: Final = sa.text("""
    INSERT INTO runtime_variants
        (name, description, reads_vfolder_config_files, default_model_definition)
    VALUES (:name, 'MLX LM', FALSE, CAST(:definition AS JSONB))
    ON CONFLICT (name) DO NOTHING
""")

_INSERT_PRESET: Final = sa.text("""
    INSERT INTO runtime_variant_presets
        (runtime_variant, name, description, rank, preset_target, value_type,
         default_value, key, category, display_name, ui_option)
    SELECT
        rv.id, :preset_name, :description, :rank, :preset_target, :value_type,
        :default_value, :key, :category, :display_name, CAST(:ui_option AS JSONB)
    FROM runtime_variants rv
    WHERE rv.name = :name
    ON CONFLICT ON CONSTRAINT uq_runtime_variant_presets_variant_name DO NOTHING
""")

# The variant row and every preset row of it, as (entity_type, entity_id).
_ENTITIES: Final[str] = """
    SELECT 'runtime_variant' AS entity_type, rv.id AS entity_id
    FROM runtime_variants rv WHERE rv.name = :name
    UNION ALL
    SELECT 'runtime_variant_preset', p.id
    FROM runtime_variant_presets p
    JOIN runtime_variants rv ON rv.id = p.runtime_variant
    WHERE rv.name = :name
"""

_NODES: Final[str] = f"""
    SELECT ve.id FROM virtual_entities ve
    JOIN ({_ENTITIES}) e ON ve.entity_type = e.entity_type AND ve.entity_id = e.entity_id
"""

# The scopes a runtime variant and its presets are created in.
_SCOPES: Final[str] = """
    SELECT ve.id FROM virtual_entities ve
    JOIN global_entities ge ON ve.entity_type = 'global' AND ve.entity_id = ge.id
    WHERE ge.name IN ('global', 'public')
"""

_INSERT_NODES: Final = sa.text(f"""
    INSERT INTO virtual_entities (entity_type, entity_id)
    {_ENTITIES}
    ON CONFLICT (entity_type, entity_id) DO NOTHING
""")

# Each node owns and governs itself, and each scope owns and governs it.
_INSERT_MEMBERSHIPS: Final = sa.text(f"""
    INSERT INTO entity_memberships (virtual_entity_id, member_entity_id, capped)
    SELECT n.id, n.id, FALSE FROM ({_NODES}) n
    UNION ALL
    SELECT s.id, n.id, FALSE FROM ({_SCOPES}) s, ({_NODES}) n
    ON CONFLICT (virtual_entity_id, member_entity_id) DO NOTHING
""")

_INSERT_BINDINGS: Final = sa.text(f"""
    INSERT INTO scope_bindings (virtual_entity_id, scope_entity_id)
    SELECT n.id, n.id FROM ({_NODES}) n
    UNION ALL
    SELECT n.id, s.id FROM ({_SCOPES}) s, ({_NODES}) n
    ON CONFLICT DO NOTHING
""")

# Edges go with the node: both edge tables cascade on its deletion.
_DELETE_PRESET_NODES: Final = sa.text("""
    DELETE FROM virtual_entities ve
    USING runtime_variant_presets p, runtime_variants rv
    WHERE ve.entity_type = 'runtime_variant_preset' AND ve.entity_id = p.id
      AND p.runtime_variant = rv.id AND rv.name = :name
      AND p.name = ANY(:preset_names)
""")

_DELETE_PRESETS: Final = sa.text("""
    DELETE FROM runtime_variant_presets p
    USING runtime_variants rv
    WHERE p.runtime_variant = rv.id AND rv.name = :name
      AND p.name = ANY(:preset_names)
""")

# A variant a deployment revision or a remaining preset still names is kept.
_UNUSED_VARIANT: Final[str] = """
    rv.name = :name
    AND NOT EXISTS (SELECT 1 FROM deployment_revisions dr WHERE dr.runtime_variant_id = rv.id)
    AND NOT EXISTS (SELECT 1 FROM runtime_variant_presets p WHERE p.runtime_variant = rv.id)
"""

_DELETE_VARIANT_NODE: Final = sa.text(f"""
    DELETE FROM virtual_entities ve
    USING runtime_variants rv
    WHERE ve.entity_type = 'runtime_variant' AND ve.entity_id = rv.id
      AND {_UNUSED_VARIANT}
""")

_DELETE_VARIANT: Final = sa.text(f"DELETE FROM runtime_variants rv WHERE {_UNUSED_VARIANT}")


def seed(conn: sa.Connection) -> None:
    conn.execute(
        _INSERT_VARIANT,
        {"name": VARIANT_NAME, "definition": json.dumps(DEFAULT_MODEL_DEFINITION)},
    )
    for preset in PRESETS:
        conn.execute(
            _INSERT_PRESET,
            {
                "name": VARIANT_NAME,
                "preset_name": preset["name"],
                "description": preset["description"],
                "rank": preset["rank"],
                "preset_target": preset["preset_target"],
                "value_type": preset["value_type"],
                "default_value": preset["default_value"],
                "key": preset["key"],
                "category": preset["category"],
                "display_name": preset["display_name"],
                "ui_option": json.dumps(preset["ui_option"]),
            },
        )
    params = {"name": VARIANT_NAME}
    conn.execute(_INSERT_NODES, params)
    conn.execute(_INSERT_MEMBERSHIPS, params)
    conn.execute(_INSERT_BINDINGS, params)


def remove_seed(conn: sa.Connection) -> None:
    params = {"name": VARIANT_NAME, "preset_names": [preset["name"] for preset in PRESETS]}
    conn.execute(_DELETE_PRESET_NODES, params)
    conn.execute(_DELETE_PRESETS, params)
    conn.execute(_DELETE_VARIANT_NODE, {"name": VARIANT_NAME})
    conn.execute(_DELETE_VARIANT, {"name": VARIANT_NAME})


def upgrade() -> None:
    seed(op.get_bind())


def downgrade() -> None:
    remove_seed(op.get_bind())
