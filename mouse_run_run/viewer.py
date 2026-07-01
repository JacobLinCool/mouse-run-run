from __future__ import annotations

import argparse
import json
import math
import mimetypes
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import torch
from torch.nn import functional as F

from mouse_run_run.degenerate import episode_degeneracy
from mouse_run_run.env import BatchedChaseEnv, GridWorldConfig
from mouse_run_run.evaluate import OpponentMode
from mouse_run_run.policy import RNNActorCritic
from mouse_run_run.serialization import CHECKPOINT_FORMAT, load_checkpoint, read_metadata
from mouse_run_run.train import DEVICE_CHOICES, select_device


ACTION_LABELS = ("U", "R", "D", "L")
STATIC_DIR = Path(__file__).with_name("viewer_static")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args()

    server = _ViewerServer(
        (args.host, args.port),
        _ViewerHandler,
        runs_root=args.runs_root.resolve(),
        default_device=args.device,
    )
    print(f"viewer_url=http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("viewer_stopped", flush=True)


class _ViewerServer(ThreadingHTTPServer):
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


class _ViewerHandler(BaseHTTPRequestHandler):
    server: _ViewerServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_static("index.html")
                return
            if parsed.path.startswith("/static/"):
                self._send_static(parsed.path.removeprefix("/static/"))
                return
            if parsed.path == "/api/checkpoints":
                self._send_json({"checkpoints": self._checkpoints()})
                return
            if parsed.path == "/api/trajectory":
                query = parse_qs(parsed.query)
                self._send_json(self._trajectory(query))
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
        root = self.server.runs_root
        if not root.exists():
            return []
        checkpoints = []
        for path in root.rglob("*.safetensors"):
            try:
                metadata = read_metadata(path)
            except Exception:
                continue
            if metadata.get("format") != CHECKPOINT_FORMAT:
                continue
            config = _loads_metadata(metadata, "config")
            metrics = _loads_metadata(metadata, "metrics")
            stat = path.stat()
            checkpoints.append(
                {
                    "path": _display_path(path.resolve(), self.server.project_root),
                    "name": path.name,
                    "task": (config.get("env") or {}).get("task"),
                    "seed": _infer_seed(path),
                    "update": _infer_update(path),
                    "size_bytes": stat.st_size,
                    "modified_unix": stat.st_mtime,
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(stat.st_mtime),
                    ),
                    "metrics": _json_finite(metrics),
                }
            )
        checkpoints.sort(key=lambda item: float(item["modified_unix"]), reverse=True)
        return checkpoints

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


@torch.no_grad()
def generate_trajectory(
    checkpoint: Path,
    *,
    episode_seed: int = 0,
    deterministic: bool = True,
    opponent_mode: OpponentMode = "self_play",
    device_name: str = "auto",
) -> dict[str, object]:
    torch.manual_seed(episode_seed)
    device = select_device(device_name)
    config, checkpoint_metrics, chaser_state, explorer_state = load_checkpoint(checkpoint)
    env_config = GridWorldConfig(**config["env"])

    chaser = RNNActorCritic(
        env_config.observation_size,
        hidden_size=config["hidden_size"],
    ).to(device)
    explorer = RNNActorCritic(
        env_config.observation_size,
        hidden_size=config["hidden_size"],
    ).to(device)
    chaser.load_state_dict(chaser_state)
    explorer.load_state_dict(explorer_state)
    chaser.eval()
    explorer.eval()

    env = BatchedChaseEnv(env_config, batch_size=1, device=device)
    chaser_observation, explorer_observation = env.reset()
    chaser_hidden = chaser.initial_hidden(1, device)
    explorer_hidden = explorer.initial_hidden(1, device)

    chaser_return = 0.0
    explorer_return = 0.0
    collisions = 0
    chaser_new_fields = 0
    explorer_new_fields = 0
    chaser_visible_steps = 0
    explorer_visible_steps = 0
    chaser_visited = {_position_key(env.chaser_position)}
    explorer_visited = {_position_key(env.explorer_position)}
    chaser_positions = [env.chaser_position.clone()]
    explorer_positions = [env.explorer_position.clone()]
    chaser_actions = []
    explorer_actions = []
    frames = [
        _frame(
            t=0,
            env=env,
            chaser_visited=chaser_visited,
            explorer_visited=explorer_visited,
            chaser_return=chaser_return,
            explorer_return=explorer_return,
        )
    ]

    for step in range(env_config.max_steps):
        previous_chaser = _position(env.chaser_position)
        previous_explorer = _position(env.explorer_position)
        chaser_output = chaser(chaser_observation, chaser_hidden)
        explorer_output = explorer(explorer_observation, explorer_hidden)
        chaser_probs = F.softmax(chaser_output.logits, dim=-1)
        explorer_probs = F.softmax(explorer_output.logits, dim=-1)
        chaser_action = _select_action(chaser_probs, deterministic)
        explorer_action = _select_action(explorer_probs, deterministic)
        if opponent_mode == "random_chaser":
            chaser_action = torch.randint(4, (1,), device=device)
        elif opponent_mode == "random_explorer":
            explorer_action = torch.randint(4, (1,), device=device)
        elif opponent_mode != "self_play":
            raise ValueError(f"Unsupported opponent mode: {opponent_mode}")

        result = env.step(chaser_action, explorer_action)
        chaser_actions.append(chaser_action.clone())
        explorer_actions.append(explorer_action.clone())
        chaser_positions.append(result.chaser_position.clone())
        explorer_positions.append(result.explorer_position.clone())
        chaser_return += _float(result.chaser_reward)
        explorer_return += _float(result.explorer_reward)
        chaser_movement = _movement(
            before=previous_chaser,
            after=_position(result.chaser_position),
            action=int(_long(chaser_action)),
            collision=_bool(result.collision),
            grid_size=env_config.grid_size,
        )
        explorer_movement = _movement(
            before=previous_explorer,
            after=_position(result.explorer_position),
            action=int(_long(explorer_action)),
            collision=_bool(result.collision),
            grid_size=env_config.grid_size,
        )
        collisions += int(_bool(result.collision))
        chaser_new_fields += int(_bool(result.chaser_new_field))
        explorer_new_fields += int(_bool(result.explorer_new_field))
        chaser_visible_steps += int(_bool(result.chaser_partner_visible))
        explorer_visible_steps += int(_bool(result.explorer_partner_visible))
        chaser_visited.add(_position_key(result.chaser_position))
        explorer_visited.add(_position_key(result.explorer_position))

        frames.append(
            _frame(
                t=step + 1,
                env=env,
                chaser_visited=chaser_visited,
                explorer_visited=explorer_visited,
                chaser_return=chaser_return,
                explorer_return=explorer_return,
                chaser_action=int(_long(chaser_action)),
                explorer_action=int(_long(explorer_action)),
                chaser_probabilities=_probabilities(chaser_probs),
                explorer_probabilities=_probabilities(explorer_probs),
                chaser_reward=_float(result.chaser_reward),
                explorer_reward=_float(result.explorer_reward),
                collision=_bool(result.collision),
                chaser_new_field=_bool(result.chaser_new_field),
                explorer_new_field=_bool(result.explorer_new_field),
                chaser_approach=_bool(result.chaser_approach),
                explorer_escape=_bool(result.explorer_escape),
                explorer_escape_close=_bool(result.explorer_escape_close),
                explorer_escape_near=_bool(result.explorer_escape_near),
                explorer_escape_far=_bool(result.explorer_escape_far),
                chaser_movement=chaser_movement,
                explorer_movement=explorer_movement,
                chaser_subspace_norm=_float(
                    chaser.neural_action_subspace(chaser_output.hidden).norm(dim=1)
                ),
                explorer_subspace_norm=_float(
                    explorer.neural_action_subspace(explorer_output.hidden).norm(dim=1)
                ),
            )
        )

        chaser_hidden = chaser_output.hidden
        explorer_hidden = explorer_output.hidden
        chaser_observation = result.chaser_observation
        explorer_observation = result.explorer_observation

    horizon = float(env_config.max_steps)
    degeneracy = episode_degeneracy(
        torch.stack(chaser_positions),
        torch.stack(explorer_positions),
        torch.stack(chaser_actions),
        torch.stack(explorer_actions),
        threshold_fraction=0.01,
    )
    return {
        "checkpoint": str(checkpoint),
        "config": config,
        "checkpoint_metrics": _json_finite(checkpoint_metrics),
        "episode_seed": episode_seed,
        "deterministic": deterministic,
        "opponent_mode": opponent_mode,
        "device": str(device),
        "summary": {
            "collisions": collisions,
            "chaser_return": chaser_return,
            "explorer_return": explorer_return,
            "chaser_new_fields": chaser_new_fields,
            "explorer_new_fields": explorer_new_fields,
            "chaser_partner_vision": chaser_visible_steps / horizon,
            "explorer_partner_vision": explorer_visible_steps / horizon,
            "final_distance": frames[-1]["distance"],
            "degenerate": _tensor_bool(degeneracy["degenerate"]),
            "degenerate_threshold_fraction": 0.01,
            "degenerate_threshold_steps": _tensor_float(degeneracy["threshold_steps"]),
            "same_state_steps": _tensor_int(degeneracy["same_state_steps"]),
            "chaser_stationary_steps": _tensor_int(degeneracy["chaser_stationary_steps"]),
            "explorer_stationary_steps": _tensor_int(degeneracy["explorer_stationary_steps"]),
        },
        "action_labels": ACTION_LABELS,
        "frames": frames,
    }


def _frame(
    *,
    t: int,
    env: BatchedChaseEnv,
    chaser_visited: set[tuple[int, int]],
    explorer_visited: set[tuple[int, int]],
    chaser_return: float,
    explorer_return: float,
    chaser_action: int | None = None,
    explorer_action: int | None = None,
    chaser_probabilities: list[float] | None = None,
    explorer_probabilities: list[float] | None = None,
    chaser_reward: float = 0.0,
    explorer_reward: float = 0.0,
    collision: bool = False,
    chaser_new_field: bool = False,
    explorer_new_field: bool = False,
    chaser_approach: bool = False,
    explorer_escape: bool = False,
    explorer_escape_close: bool = False,
    explorer_escape_near: bool = False,
    explorer_escape_far: bool = False,
    chaser_movement: dict[str, object] | None = None,
    explorer_movement: dict[str, object] | None = None,
    chaser_subspace_norm: float = 0.0,
    explorer_subspace_norm: float = 0.0,
) -> dict[str, object]:
    chaser_position = _position(env.chaser_position)
    explorer_position = _position(env.explorer_position)
    chaser_visible = _bool(env.partner_visible())
    explorer_visible = _bool(env._partner_visible(env.explorer_position, env.chaser_position))
    return {
        "t": t,
        "chaser": chaser_position,
        "explorer": explorer_position,
        "chaser_visited": _sorted_positions(chaser_visited),
        "explorer_visited": _sorted_positions(explorer_visited),
        "distance": _float(env.distance()),
        "chaser_partner_visible": chaser_visible,
        "explorer_partner_visible": explorer_visible,
        "actions": {
            "chaser": chaser_action,
            "explorer": explorer_action,
            "chaser_label": None if chaser_action is None else ACTION_LABELS[chaser_action],
            "explorer_label": None if explorer_action is None else ACTION_LABELS[explorer_action],
        },
        "probabilities": {
            "chaser": chaser_probabilities,
            "explorer": explorer_probabilities,
        },
        "rewards": {
            "chaser": chaser_reward,
            "explorer": explorer_reward,
        },
        "returns": {
            "chaser": chaser_return,
            "explorer": explorer_return,
        },
        "events": {
            "collision": collision,
            "chaser_new_field": chaser_new_field,
            "explorer_new_field": explorer_new_field,
            "chaser_approach": chaser_approach,
            "explorer_escape": explorer_escape,
            "explorer_escape_close": explorer_escape_close,
            "explorer_escape_near": explorer_escape_near,
            "explorer_escape_far": explorer_escape_far,
        },
        "movement": {
            "chaser": chaser_movement or _empty_movement(),
            "explorer": explorer_movement or _empty_movement(),
        },
        "subspace_norm": {
            "chaser": chaser_subspace_norm,
            "explorer": explorer_subspace_norm,
        },
    }


def _select_action(probabilities: torch.Tensor, deterministic: bool) -> torch.Tensor:
    if deterministic:
        return probabilities.argmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(1)


def _movement(
    *,
    before: list[int],
    after: list[int],
    action: int,
    collision: bool,
    grid_size: int,
) -> dict[str, object]:
    row_delta, col_delta = ((-1, 0), (0, 1), (1, 0), (0, -1))[action]
    raw = [before[0] + row_delta, before[1] + col_delta]
    moved = before != after
    blocked_by_wall = (
        raw[0] < 0
        or raw[0] >= grid_size
        or raw[1] < 0
        or raw[1] >= grid_size
    )
    blocked_by_collision = (not moved) and (not blocked_by_wall) and collision
    return {
        "moved": moved,
        "blocked": not moved,
        "blocked_by_wall": blocked_by_wall,
        "blocked_by_collision": blocked_by_collision,
        "attempted": raw,
    }


def _empty_movement() -> dict[str, object]:
    return {
        "moved": False,
        "blocked": False,
        "blocked_by_wall": False,
        "blocked_by_collision": False,
        "attempted": None,
    }


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


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _loads_metadata(metadata: dict[str, str], key: str) -> dict[str, object]:
    raw = metadata.get(key)
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(decoded, dict):
        return decoded
    return {}


def _infer_seed(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.removeprefix("seed_"))
            except ValueError:
                return None
    return None


def _infer_update(path: Path) -> int | None:
    stem = path.stem
    if stem.startswith("update_"):
        try:
            return int(stem.removeprefix("update_"))
        except ValueError:
            return None
    if stem == "latest" or stem == "final":
        return None
    return None


def _first(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key)
    if not values:
        return default
    return values[0]


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _position(tensor: torch.Tensor) -> list[int]:
    return [int(tensor[0, 0].item()), int(tensor[0, 1].item())]


def _position_key(tensor: torch.Tensor) -> tuple[int, int]:
    return (int(tensor[0, 0].item()), int(tensor[0, 1].item()))


def _sorted_positions(values: set[tuple[int, int]]) -> list[list[int]]:
    return [[row, col] for row, col in sorted(values)]


def _probabilities(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor[0].detach().cpu().tolist()]


def _bool(tensor: torch.Tensor) -> bool:
    return bool(tensor[0].item())


def _float(tensor: torch.Tensor) -> float:
    return float(tensor[0].item())


def _long(tensor: torch.Tensor) -> int:
    return int(tensor[0].item())


def _tensor_bool(tensor: torch.Tensor) -> bool:
    return bool(tensor.reshape(-1)[0].item())


def _tensor_int(tensor: torch.Tensor) -> int:
    return int(tensor.reshape(-1)[0].item())


def _tensor_float(tensor: torch.Tensor) -> float:
    return float(tensor.reshape(-1)[0].item())


def _json_finite(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
