from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


REQUIRED = {
    "forgemind-paper.pdf",
    "summary.json",
    "demo-script.md",
    "SOURCE_REVISION",
}
FORBIDDEN_PARTS = {"models", "private", "raw", "logs", "__pycache__"}
CHECKSUMS_NAME = "SHA256SUMS.json"
ZIP_TIMESTAMP = (2026, 8, 7, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_members(members: list[str]) -> list[str]:
    errors: list[str] = []
    normalized: list[str] = []
    for member in members:
        posix = member.replace("\\", "/")
        path = PurePosixPath(posix)
        windows_path = PureWindowsPath(member)
        if path.is_absolute() or windows_path.is_absolute() or ".." in path.parts:
            errors.append(f"unsafe submission member: {member}")
            continue
        normalized.append(path.as_posix())
        lowered = [part.lower() for part in path.parts]
        if (
            any(part in FORBIDDEN_PARTS for part in lowered)
            or any(part == ".env" or part.startswith(".env.") for part in lowered)
            or path.suffix.lower() == ".gguf"
        ):
            errors.append(f"forbidden submission member: {member}")

    for missing in sorted(REQUIRED - set(normalized)):
        errors.append(f"missing required file: {missing}")
    if len(normalized) != len(set(normalized)):
        errors.append("duplicate submission member")
    return errors


def build_bundle(staging: Path, output: Path) -> None:
    staging = staging.resolve()
    output = output.resolve()
    if not staging.is_dir():
        raise ValueError(f"staging directory not found: {staging}")
    try:
        output.relative_to(staging)
    except ValueError:
        pass
    else:
        raise ValueError("submission output must be outside the staging directory")

    symlinks = [path for path in staging.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"submission staging contains symlink: {symlinks[0]}")

    members = sorted(
        (
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != CHECKSUMS_NAME
        ),
        key=lambda path: path.relative_to(staging).as_posix(),
    )
    relative = [path.relative_to(staging).as_posix() for path in members]
    errors = validate_members(relative)
    if errors:
        raise ValueError("; ".join(errors))

    checksums = {name: sha256_file(staging / name) for name in relative}
    checksum_path = staging / CHECKSUMS_NAME
    checksum_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    members = sorted(
        (path for path in staging.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(staging).as_posix(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for path in members:
            info = zipfile.ZipInfo(path.relative_to(staging).as_posix(), ZIP_TIMESTAMP)
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, policy-checked ForgeMind submission bundle."
    )
    parser.add_argument("staging", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    args.staging.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (args.staging / "SOURCE_REVISION").write_text(
        revision + "\n", encoding="utf-8"
    )
    build_bundle(args.staging, args.output)
    print(f"sha256 {sha256_file(args.output)}  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
