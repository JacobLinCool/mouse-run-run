import json

import pytest

from mouse_run_run.benchmark import parse_positive_ints, steady_timing


def test_steady_timing_excludes_first_update(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"update": 1, "elapsed_seconds": 10.0},
        {"update": 2, "elapsed_seconds": 14.0},
        {"update": 3, "elapsed_seconds": 19.0},
        # Final completed snapshots may repeat the last update.
        {"update": 3, "elapsed_seconds": 19.2},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = steady_timing(path, expected_updates=3)

    assert result["steady_elapsed_seconds"] == 9.0
    assert result["steady_update_seconds"] == 4.5


def test_steady_timing_rejects_incomplete_evidence(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"update": 1, "elapsed_seconds": 1.0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete timing records"):
        steady_timing(path, expected_updates=2)


def test_concurrency_parser_requires_positive_values() -> None:
    assert parse_positive_ints("1,2,10") == [1, 2, 10]
    with pytest.raises(ValueError, match="positive integers"):
        parse_positive_ints("1,0")
