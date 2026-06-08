import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_manifest(
    job_id: str,
    engine: str,
    selected_profile: str,
    input_file: str | None = None,
    outputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    steps: list[str] | None = None,
    model: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "selected_profile": selected_profile,
        "input_file": input_file,
        "steps": steps or [],
        "model": model or {},
        "outputs": outputs or {},
        "metrics": metrics or {},
        "errors": errors or [],
    }


def write_manifest(manifest: dict[str, Any], output_dir: str | Path) -> str:
    output_path = Path(output_dir) / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return str(output_path)