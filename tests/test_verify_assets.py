import hashlib
import json
from pathlib import Path

from scripts.verify_assets import verify_manifest


def test_verify_manifest_accepts_matching_asset(tmp_path: Path) -> None:
    payload = b"model fixture"
    (tmp_path / "model.gguf").write_bytes(payload)
    manifest = {
        "assets": [
            {
                "path": "model.gguf",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source": "https://example.invalid/model",
                "license": "test-only",
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_manifest(path, tmp_path) == []


def test_verify_manifest_reports_missing_and_changed_assets(tmp_path: Path) -> None:
    manifest = {
        "assets": [
            {
                "path": "missing.bin",
                "sha256": "0" * 64,
                "source": "x",
                "license": "x",
            },
            {
                "path": "changed.bin",
                "sha256": "1" * 64,
                "source": "x",
                "license": "x",
            },
        ]
    }
    (tmp_path / "changed.bin").write_bytes(b"changed")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = verify_manifest(path, tmp_path)

    assert any("missing.bin: missing" in error for error in errors)
    assert any("changed.bin: checksum mismatch" in error for error in errors)


def test_verify_manifest_rejects_paths_outside_asset_root(tmp_path: Path) -> None:
    manifest = {
        "assets": [
            {
                "path": "../secret.bin",
                "sha256": "0" * 64,
                "source": "x",
                "license": "x",
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_manifest(path, tmp_path) == ["../secret.bin: escapes asset root"]
