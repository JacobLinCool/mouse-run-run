from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi

from mouse_run_run.provenance import collect_provenance, hash_file, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("dataset", "model", "space"), default="dataset")
    parser.add_argument("--path-in-repo", default="")
    parser.add_argument("--revision")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload mouse-run-run experiment artifacts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    artifact_path = args.path.expanduser()
    if not artifact_path.exists():
        raise FileNotFoundError(str(artifact_path))

    manifest_path = args.manifest or artifact_path / "HF_UPLOAD_MANIFEST.json"
    files = sorted(
        path
        for path in artifact_path.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_path": str(artifact_path),
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "path_in_repo": args.path_in_repo,
        "revision": args.revision,
        "private": args.private,
        "file_count": len(files),
        "excluded_from_file_manifest": [str(manifest_path)],
        "files": [
            {
                "path": str(path.relative_to(artifact_path)),
                "size_bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in files
        ],
        "provenance": collect_provenance(cwd=Path.cwd()),
    }
    write_json_atomic(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(artifact_path),
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
    )
    print(f"uploaded={artifact_path} repo={args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
