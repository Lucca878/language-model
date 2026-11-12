"""Small helpers for project config, prompt templates, and response schemas."""

import json
from pathlib import Path
from string import Template
from typing import Any

import yaml
from dotenv import load_dotenv


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    """Load the config and .env from the directory where the app is run."""
    load_dotenv(".env")
    with Path(config_path).expanduser().open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def resolve_path(value: str) -> Path:
    """Interpret a path relative to the current working directory."""
    return Path(value).expanduser()


def build_prompt(
    task: dict[str, Any],
    row: dict[str, Any],
    statement_id: Any,
    statement_text: str,
    label: Any,
    project_name: str,
) -> str:
    """Render a task prompt by substituting values into its template file."""
    prompt_config = task.get("prompt", {})
    context = {
        **row,
        **prompt_config.get("variables", {}),
        "statement_id": statement_id,
        "statement_text": statement_text,
        "text": statement_text,
        "label": label,
        "project_name": project_name,
        "task_name": task["name"],
        "row_json": json.dumps(row, ensure_ascii=False, default=str),
    }

    template = resolve_path(prompt_config["template_file"]).read_text(encoding="utf-8")
    return Template(template).substitute(context)


def load_response_schema(task: dict[str, Any]) -> dict[str, Any] | None:
    """Load a schema from a JSON file or inline config."""
    schema_config = task.get("schema")
    if not schema_config:
        return None
    if "file" in schema_config:
        with resolve_path(schema_config["file"]).open(encoding="utf-8") as schema_file:
            return json.load(schema_file)
    return schema_config.get("inline")