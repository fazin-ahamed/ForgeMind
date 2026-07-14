from pathlib import Path

from forgemind.ingest import discover_text_sources


def test_discovery_skips_binary_secret_vendor_and_escaping_symlink(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('safe')\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-12345678901234567890\n", encoding="utf-8"
    )
    (tmp_path / "image.bin").write_bytes(b"abc\x00def")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("vendor", encoding="utf-8")
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("must not ingest", encoding="utf-8")
    try:
        (tmp_path / "escape.txt").symlink_to(outside)
    except OSError:
        pass

    sources = discover_text_sources(tmp_path)

    assert [source.path for source in sources] == ["src/app.py"]
