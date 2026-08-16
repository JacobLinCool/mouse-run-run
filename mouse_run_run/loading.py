"""Strict loading of typed Python experiment definitions."""

from __future__ import annotations

import importlib
import json
from dataclasses import fields, is_dataclass, replace
from typing import Any

from mouse_run_run.core.experiment import Experiment, ExperimentDefinition
from mouse_run_run.studies.types import StudyDefinition


def load_experiment(target: str, overrides: list[str] | None = None) -> Experiment:
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("experiment target must be module.path:attribute")
    module = importlib.import_module(module_name)
    try:
        value = getattr(module, attribute_name)
    except AttributeError:
        raise ValueError(f"experiment target {target!r} does not exist") from None
    assignments = _parse_overrides(overrides or [])
    if isinstance(value, ExperimentDefinition):
        config = value.config
        for path, replacement in assignments:
            config = _replace_path(config, path, replacement)
        return value.resolve(config)
    if isinstance(value, Experiment):
        if assignments:
            raise ValueError(
                "--set requires an ExperimentDefinition target with an editable config"
            )
        value.validate()
        return value
    raise TypeError(
        f"experiment target {target!r} must be ExperimentDefinition or Experiment, "
        f"got {type(value)!r}"
    )


def load_study(target: str) -> StudyDefinition:
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("study target must be module.path:attribute")
    module = importlib.import_module(module_name)
    try:
        value = getattr(module, attribute_name)
    except AttributeError:
        raise ValueError(f"study target {target!r} does not exist") from None
    if not isinstance(value, StudyDefinition):
        raise TypeError(
            f"study target {target!r} must be StudyDefinition, got {type(value)!r}"
        )
    value.validate()
    return value


def _parse_overrides(values: list[str]) -> list[tuple[tuple[str, ...], Any]]:
    parsed = []
    for value in values:
        path, separator, raw = value.partition("=")
        if not separator or not path or any(not part.isidentifier() for part in path.split(".")):
            raise ValueError(f"invalid override {value!r}; expected path.to.field=JSON_VALUE")
        try:
            replacement = json.loads(raw)
        except json.JSONDecodeError:
            replacement = raw
        parsed.append((tuple(path.split(".")), replacement))
    return parsed


def _replace_path(value: Any, path: tuple[str, ...], replacement: Any) -> Any:
    if not path:
        return _coerce_like(value, replacement)
    if not is_dataclass(value):
        raise ValueError(f"override path enters non-dataclass value at {'.'.join(path)!r}")
    field_names = {field.name for field in fields(value)}
    head, *tail = path
    if head not in field_names:
        raise ValueError(f"unknown override field {head!r} on {type(value).__name__}")
    current = getattr(value, head)
    changed = _replace_path(current, tuple(tail), replacement)
    return replace(value, **{head: changed})


def _coerce_like(current: Any, replacement: Any) -> Any:
    if isinstance(current, tuple) and isinstance(replacement, list):
        return tuple(replacement)
    if isinstance(current, bool) and not isinstance(replacement, bool):
        raise ValueError("boolean overrides must be JSON true or false")
    if isinstance(current, int) and not isinstance(current, bool):
        if not isinstance(replacement, int) or isinstance(replacement, bool):
            raise ValueError("integer override requires a JSON integer")
    if isinstance(current, float) and not isinstance(replacement, int | float):
        raise ValueError("float override requires a JSON number")
    return replacement
