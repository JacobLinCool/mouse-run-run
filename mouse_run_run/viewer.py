from __future__ import annotations

import argparse
from pathlib import Path

from mouse_run_run.train import DEVICE_CHOICES
from mouse_run_run.viewer_http import _ViewerHandler, _ViewerServer


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


if __name__ == "__main__":
    main()
