"""Save each LLM run as both CSV and JSON."""

import csv
import json
from pathlib import Path
from typing import Any


def _csv_value(record: dict[str, Any], field_path: str) -> Any:
    value: Any = record
    for key in field_path.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def store_records(
    records: list[dict[str, Any]],
    output_path: Path,
    csv_columns: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write records to sibling CSV and JSON files and return both paths."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_name(f"{output_path.name}.csv")
    json_path = output_path.with_name(f"{output_path.name}.json")

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=2, default=str)

    if csv_columns:
        columns = list(csv_columns)
        rows = [
            {
                column: _csv_value(record, field_path)
                for column, field_path in csv_columns.items()
            }
            for record in records
        ]
    else:
        rows = [
            {
                **record["metadata"],
                "payload": json.dumps(record["payload"], ensure_ascii=False, default=str),
            }
            for record in records
        ]
        columns = list(dict.fromkeys(column for row in rows for column in row))
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, json_path
