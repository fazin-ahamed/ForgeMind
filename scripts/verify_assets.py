from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path, root: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset_root = root.resolve()
    errors: list[str] = []
    for asset in manifest["assets"]:
        relative = str(asset["path"])
        path = (asset_root / relative).resolve()
        try:
            path.relative_to(asset_root)
        except ValueError:
            errors.append(f"{relative}: escapes asset root")
            continue
        if not path.is_file():
            errors.append(f"{relative}: missing")
        elif sha256_file(path) != str(asset["sha256"]).lower():
            errors.append(f"{relative}: checksum mismatch")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify preloaded ForgeMind assets without downloading them."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    errors = verify_manifest(args.manifest, args.root)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
