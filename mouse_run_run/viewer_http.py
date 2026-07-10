from __future__ import annotations

import json
import mimetypes
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from mouse_run_run.training_config import DEVICE_CHOICES
from mouse_run_run.viewer_payloads import (
    build_report,
    checkpoint_listing,
    compute_shared_subspace,
    generate_trajectory,
)


STATIC_DIR = Path(__file__).with_name("viewer_static")


class ViewerServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        runs_root: Path,
        default_device: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.runs_root = runs_root
        self.default_device = default_device
        self.project_root = Path.cwd().resolve()


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_static("index.html")
                return
            if parsed.path == "/report":
                self._send_static("report.html")
                return
            if parsed.path.startswith("/static/"):
                self._send_static(parsed.path.removeprefix("/static/"))
                return
            if parsed.path == "/api/checkpoints":
                self._send_json({"checkpoints": self._checkpoints()})
                return
            if parsed.path == "/api/report":
                self._send_json(self._report())
                return
            if parsed.path == "/api/trajectory":
                query = parse_qs(parsed.query)
                self._send_json(self._trajectory(query))
                return
            if parsed.path == "/api/shared_subspace":
                query = parse_qs(parsed.query)
                self._send_json(self._shared_subspace(query))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001 - server should return JSON errors.
            self._send_json(
                {"error": type(exc).__name__, "message": str(exc)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: object) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {format % args}", flush=True)

    def _send_static(self, relative_path: str) -> None:
        path = (STATIC_DIR / relative_path).resolve()
        if STATIC_DIR.resolve() not in path.parents and path != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, value: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, allow_nan=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _checkpoints(self) -> list[dict[str, object]]:
        return checkpoint_listing(self.server.runs_root, self.server.project_root)

    def _report(self) -> dict[str, object]:
        return build_report(self.server.runs_root, self.server.project_root)

    def _trajectory(self, query: dict[str, list[str]]) -> dict[str, object]:
        checkpoint_raw = _first(query, "checkpoint")
        if not checkpoint_raw:
            raise ValueError("checkpoint is required")
        checkpoint = _resolve_checkpoint(
            unquote(checkpoint_raw),
            project_root=self.server.project_root,
        )
        episode_seed = int(_first(query, "seed", "0"))
        deterministic = _parse_bool(_first(query, "deterministic", "true"))
        opponent_mode = _first(query, "opponent", "self_play")
        device = _first(query, "device", self.server.default_device)
        filter_degenerate = _parse_bool(_first(query, "filter_degenerate", "false"))
        if opponent_mode not in ("self_play", "random_chaser", "random_explorer"):
            raise ValueError(f"Unsupported opponent mode: {opponent_mode}")
        if device not in DEVICE_CHOICES:
            raise ValueError(f"Unsupported device: {device}")
        attempts = 20 if filter_degenerate else 1
        last_trajectory = None
        for attempt in range(attempts):
            trajectory = generate_trajectory(
                checkpoint,
                episode_seed=episode_seed + attempt,
                deterministic=deterministic,
                opponent_mode=opponent_mode,  # type: ignore[arg-type]
                device_name=device,
            )
            trajectory["filter"] = {
                "filter_degenerate": filter_degenerate,
                "requested_seed": episode_seed,
                "attempts": attempt + 1,
                "accepted": not trajectory["summary"]["degenerate"],
            }
            last_trajectory = trajectory
            if not filter_degenerate or not trajectory["summary"]["degenerate"]:
                return trajectory
        assert last_trajectory is not None
        return last_trajectory

    def _shared_subspace(self, query: dict[str, list[str]]) -> dict[str, object]:
        checkpoint_raw = _first(query, "checkpoint")
        if not checkpoint_raw:
            raise ValueError("checkpoint is required")
        checkpoint = _resolve_checkpoint(
            unquote(checkpoint_raw),
            project_root=self.server.project_root,
        )
        device = _first(query, "device", self.server.default_device)
        if device not in DEVICE_CHOICES:
            raise ValueError(f"Unsupported device: {device}")
        episodes = _clamp_int(_first(query, "episodes", "8"), low=2, high=64)
        max_steps = _clamp_int(_first(query, "max_steps", "0"), low=0, high=1000)
        permutations = _clamp_int(_first(query, "permutations", "100"), low=20, high=1000)
        seed = int(_first(query, "seed", "0"))
        return compute_shared_subspace(
            checkpoint,
            episodes=episodes,
            max_steps=max_steps or None,
            permutations=permutations,
            seed=seed,
            device_name=device,
        )


def _first(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key)
    if not values:
        return default
    return values[0]


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _clamp_int(raw: str, *, low: int, high: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = low
    return max(low, min(high, value))


def _resolve_checkpoint(raw: str, *, project_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix != ".safetensors":
        raise ValueError("checkpoint must be a .safetensors file")
    return path


