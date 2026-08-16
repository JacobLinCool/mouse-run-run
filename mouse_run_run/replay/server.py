"""Dependency-free local server for the bundled saved-rollout viewer."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mouse_run_run.artifacts.rollout import RolloutArtifact, load_rollout
from mouse_run_run.core.experiment import Experiment
from mouse_run_run.replay.payload import build_replay_payload


class ReplayServer(ThreadingHTTPServer):
    primary: RolloutArtifact
    comparisons: list[RolloutArtifact]
    experiment: Experiment


class ReplayHandler(BaseHTTPRequestHandler):
    server: ReplayServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(
                    files("mouse_run_run").joinpath("replay_static/index.html").read_text()
                )
                return
            if parsed.path == "/api/meta":
                self._send_json(_metadata(self.server))
                return
            if parsed.path == "/api/replay":
                query = parse_qs(parsed.query)
                episode = int(query.get("episode", ["0"])[0])
                payload = {
                    "primary": build_replay_payload(
                        self.server.primary,
                        self.server.experiment,
                        episode=episode,
                    ),
                    **{
                        f"comparison_{index + 1}": build_replay_payload(
                            comparison,
                            self.server.experiment,
                            episode=episode,
                        )
                        for index, comparison in enumerate(self.server.comparisons)
                    },
                }
                self._send_json(payload)
                return
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, IndexError) as error:
            self._send_json(
                {"error": type(error).__name__, "message": str(error)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, value: str) -> None:
        payload = value.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, value: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve_replay(
    rollout: Path,
    *,
    compare: list[Path],
    experiment: Experiment,
    host: str,
    port: int,
) -> None:
    primary = load_rollout(rollout)
    comparisons = [load_rollout(path) for path in compare]
    for comparison in comparisons:
        if comparison.manifest["experiment"] != primary.manifest["experiment"]:
            raise ValueError("comparison rollout uses a different experiment")
        if comparison.trajectory.batch_size != primary.trajectory.batch_size:
            raise ValueError("comparison rollout must contain the same episode count")
        if (
            comparison.trajectory.horizon != primary.trajectory.horizon
            or comparison.trajectory.agent_ids != primary.trajectory.agent_ids
        ):
            raise ValueError("comparison rollout must have matching agents and horizon")
    server = ReplayServer((host, port), ReplayHandler)
    server.primary = primary
    server.comparisons = comparisons
    server.experiment = experiment
    print(f"replay=http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _metadata(server: ReplayServer) -> dict[str, object]:
    return {
        "format": "mrr-replay-meta-v2",
        "experiment": server.primary.manifest["experiment"],
        "episodes": server.primary.trajectory.batch_size,
        "horizon": server.primary.trajectory.horizon,
        "primary": {
            "path": str(server.primary.path),
            "interventions": server.primary.manifest["interventions"],
        },
        "comparisons": [
            {
                "path": str(comparison.path),
                "interventions": comparison.manifest["interventions"],
            }
            for comparison in server.comparisons
        ],
    }
