from benchmarks.import_external import patch_paths


def test_patch_paths_extracts_gold_files() -> None:
    patch = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
    )

    assert patch_paths(patch) == ["src/auth.py"]
