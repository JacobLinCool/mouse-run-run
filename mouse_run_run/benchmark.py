import json
from pathlib import Path
from typing import Any


def steady_timing(path: Path, expected_updates: int) -> dict[str, float]:
    """Return post-warm-up timing from append-only training metric records."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    first_by_update: dict[int, dict[str, Any]] = {}
    for row in rows:
        first_by_update.setdefault(int(row["update"]), row)
    if set(first_by_update) != set(range(1, expected_updates + 1)):
        raise ValueError(f"incomplete timing records in {path}: {sorted(first_by_update)}")
    steady_elapsed = (
        float(first_by_update[expected_updates]["elapsed_seconds"])
        - float(first_by_update[1]["elapsed_seconds"])
    )
    return {
        "steady_elapsed_seconds": steady_elapsed,
        "steady_update_seconds": steady_elapsed / (expected_updates - 1),
    }


def parse_positive_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("concurrencies must contain positive integers")
    return values
