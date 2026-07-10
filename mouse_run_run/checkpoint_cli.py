from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import SafetensorError

from mouse_run_run.health import checkpoint_health
from mouse_run_run.serialization import CHECKPOINT_FORMAT, read_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="*", default=[Path("runs")])
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    records = [checkpoint_health(path) for path in _checkpoint_paths(args.paths)]
    payload = {
        "schema_version": 1,
        "checkpoint_count": len(records),
        "healthy_checkpoint_count": sum(1 for record in records if record.ok),
        "records": [record.to_dict() for record in records],
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["healthy_checkpoint_count"] != payload["checkpoint_count"]:
        raise SystemExit(1)


def _checkpoint_paths(paths: list[Path]) -> list[Path]:
    checkpoints: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            checkpoints.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(str(path))
        for candidate in sorted(path.rglob("*.safetensors")):
            try:
                metadata = read_metadata(candidate)
            except (OSError, ValueError, SafetensorError):
                continue
            if metadata.get("format") == CHECKPOINT_FORMAT:
                checkpoints.append(candidate)
    return sorted(dict.fromkeys(checkpoints))


if __name__ == "__main__":
    main()
