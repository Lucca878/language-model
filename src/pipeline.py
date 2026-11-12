"""Generic, configuration-driven LLM project runner and result writers."""

import datetime as dt
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_client import call_llm
from project_config import (
    build_prompt,
    load_response_schema,
    resolve_path,
)
from storage import store_records

# Calls time in UTC and returns it as an ISO 8601 string
def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _output_path(
    task: dict[str, Any],
    config: dict[str, Any],
    timestamp: str,
    test_mode: bool,
) -> tuple[dict[str, Any], Path]:
    output_config = config.get("storage", {})
    filename = f"{task['name']}_{timestamp}"
    path = resolve_path(f"{output_config.get('directory', 'results')}/{filename}")
    if test_mode:
        path = path.parent / "test" / path.name
    return output_config, path


def _run_task(
    task: dict[str, Any],
    config: dict[str, Any],
    frame: pd.DataFrame,
    repeat_count: int,
    temperature_rng: np.random.Generator,
    timestamp: str,
    test_mode: bool,
) -> tuple[Path, Path]:
    """Run one task over all selected rows and save its responses."""
    project = config.get("project", {})
    dataset_config = config["dataset"]
    llm_config = {**config["llm"], **task.get("llm", {})}
    model_id = llm_config["model"]
    model_name = llm_config.get("name", model_id)
    temperature_config = llm_config.get("temperature", 0.7)
    schema = load_response_schema(task)
    response_format = task.get("response_format", llm_config.get("response_format", "json_schema"))
    output_config, output_path = _output_path(
        task, config, timestamp, test_mode
    )
    include_prompt = output_config.get("include_prompt", True)
    records: list[dict[str, Any]] = []
    total = len(frame) * repeat_count
    text_column = dataset_config.get("text_column", "text")

    print(f"\n=== {project.get('name', 'LLM project')} / {task['name']} ===")
    for statement_index, (_, source_row) in enumerate(frame.iterrows(), start=1):
        row = source_row.to_dict()
        statement_text = str(row.get(text_column, ""))
        statement_id = row.get(dataset_config.get("id_column"), statement_index)
        label = row.get(dataset_config.get("label_column"), "")
        prompt = build_prompt(
            task,
            row,
            statement_id,
            statement_text,
            label,
            str(project.get("name", "LLM project")),
        )
        input_group_id = str(uuid.uuid4())

        for repetition in range(1, repeat_count + 1):
            if isinstance(temperature_config, dict):
                temperature = float(
                    temperature_rng.uniform(
                        float(temperature_config["min"]),
                        float(temperature_config["max"]),
                    )
                )
            else:
                temperature = float(temperature_config)

            started_at = _utc_now()
            start = time.perf_counter()
            payload = call_llm(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=llm_config.get("max_tokens"),
                schema=schema,
                schema_name=task.get("schema_name", task["name"]),
                response_format=response_format,
            )
            received_at = _utc_now()
            duration_ms = round((time.perf_counter() - start) * 1000)
            metadata = {
                "project": project.get("name", "LLM project"),
                "task": task["name"],
                "input_group_id": input_group_id,
                "session_id": str(uuid.uuid4()),
                "statement_id": statement_id,
                "original_text": statement_text,
                "original_label": label,
                "annotation_id": str(uuid.uuid4()),
                "replicate_number": repetition,
                "model": model_name,
                "model_id": model_id,
                "temperature": temperature,
                "started_at": started_at,
                "received_at": received_at,
                "duration_ms": duration_ms,
                "prompt": prompt,
            }
            if not include_prompt:
                metadata.pop("prompt")
            records.append({"metadata": metadata, "input": row, "payload": payload})
            print(
                f"[{task['name']}] {len(records)}/{total} | "
                f"statement={statement_index}/{len(frame)} | "
                f"repetition={repetition}/{repeat_count} | {duration_ms}ms"
            )

    csv_path, json_path = store_records(
        records,
        output_path,
        csv_columns=output_config.get("csv_columns"),
    )
    print(f"Saved {len(records)} results -> {csv_path} and {json_path}")
    return csv_path, json_path


def run_project(
    config: dict[str, Any],
    n_statements: int | None = None,
    repetitions: int | None = None,
    test_mode: bool = False,
) -> tuple[Path, Path]:
    """Run the configured task over a sampled dataset and save its responses."""
    project = config.get("project", {})
    dataset_config = config["dataset"]
    dataset_path = resolve_path(dataset_config["path"])
    frame = pd.read_csv(dataset_path)
    text_column = dataset_config.get("text_column", "text")
    if text_column not in frame.columns:
        raise ValueError(f"Dataset has no configured text column '{text_column}'.")

    sampling = config.get("sampling", {})
    seed = int(sampling.get("seed", 42))
    requested_count = n_statements if n_statements is not None else sampling.get("count")
    if requested_count is not None and int(requested_count) < 1:
        raise ValueError("The requested sample size must be at least 1.")
    if requested_count is not None and int(requested_count) < len(frame):
        frame = frame.sample(n=int(requested_count), random_state=seed)
    frame = frame.reset_index(drop=True)

    repeat_count = repetitions if repetitions is not None else int(
        config.get("run", {}).get("repetitions", 1)
    )
    if repeat_count < 1:
        raise ValueError("Repetitions must be at least 1.")

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temperature_rng = np.random.default_rng(seed)
    output_paths = _run_task(
        config["task"],
        config,
        frame,
        repeat_count,
        temperature_rng,
        timestamp,
        test_mode,
    )
    print("\nDone.")
    return output_paths