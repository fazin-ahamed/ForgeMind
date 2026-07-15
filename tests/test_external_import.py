import gzip
import io
import json
from pathlib import Path

import benchmarks.import_external as external
from benchmarks.import_external import (
    longmemeval_candidates,
    patch_paths,
    repoqa_candidates,
)


def test_patch_paths_extracts_gold_files() -> None:
    patch = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n"
        "+++ b/src/auth.py\n"
    )

    assert patch_paths(patch) == ["src/auth.py"]


def test_repoqa_candidate_uses_description_name_and_exact_lines(
    tmp_path: Path,
) -> None:
    source = {
        "python": [
            {
                "repo": "owner/project",
                "commit_sha": "abc123",
                "content": {
                    "src/auth.py": (
                        "def helper():\n"
                        "    pass\n"
                        "def validate_session(value):\n"
                        "    return bool(value)\n"
                    )
                },
                "needles": [
                    {
                        "name": "validate_session",
                        "path": "src/auth.py",
                        "start_line": 2,
                        "end_line": 4,
                        "description": (
                            "Checks whether a supplied session value is usable."
                        ),
                    },
                    {
                        "name": "helper",
                        "path": "src/auth.py",
                        "start_line": 0,
                        "end_line": 2,
                        "description": "Performs the supporting operation.",
                    },
                ],
            }
        ]
    }

    runtime, gold = repoqa_candidates(source, tmp_path)
    validate = next(
        item for item in gold if item.answer and item.answer.accepted == ["validate_session"]
    )

    assert runtime[0].question.startswith("Which function")
    assert validate.answer is not None
    assert validate.answer.case_sensitive is True
    assert validate.evidence[0].start_line == 3
    assert validate.evidence[0].end_line == 4
    assert runtime[0].archive_path == runtime[1].archive_path


def test_longmemeval_candidate_preserves_dates_and_answer_turns(
    tmp_path: Path,
) -> None:
    source = [
        {
            "question_id": "q1",
            "question_type": "knowledge-update",
            "question": "Which city do I live in now?",
            "answer": "Dubai",
            "question_date": "2026/07/01",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2026/06/01"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "I moved to Dubai.",
                        "has_answer": True,
                    },
                    {"role": "assistant", "content": "Noted."},
                ]
            ],
        }
    ]

    runtime, gold = longmemeval_candidates(source, tmp_path)

    assert runtime[0].question == "Which city do I live in now?"
    assert gold[0].answer is not None
    assert gold[0].answer.accepted == ["Dubai"]
    session = Path(runtime[0].archive_path) / "session-0000.md"
    session_text = session.read_text(encoding="utf-8")
    assert "Session date: 2026/06/01" in session_text
    assert "Question date: 2026/07/01" in session_text
    evidence_line = session_text.splitlines()[gold[0].evidence[0].start_line - 1]
    assert "I moved to Dubai." in evidence_line


def test_repoqa_download_is_decompressed_to_json(
    tmp_path: Path, monkeypatch
) -> None:
    source = {"python": [{"repo": "owner/project"}]}
    payload = gzip.compress(json.dumps(source).encode("utf-8"))
    monkeypatch.setattr(
        external.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(payload),
    )
    output = tmp_path / "repoqa.json"

    external.import_repoqa(output)

    assert json.loads(output.read_text(encoding="utf-8")) == source


def test_longmemeval_download_applies_stable_limit(
    tmp_path: Path, monkeypatch
) -> None:
    source = [
        {"question_id": "q2"},
        {"question_id": "q1"},
    ]
    payload = json.dumps(source).encode("utf-8")
    monkeypatch.setattr(
        external.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(payload),
    )
    output = tmp_path / "longmemeval.json"

    external.import_longmemeval(output, limit=1)

    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"question_id": "q1"}
    ]


def test_longmemeval_candidate_keeps_only_evidence_sessions(tmp_path: Path) -> None:
    source = [
        {
            "question_id": "q1",
            "question": "Where did I move?",
            "answer": "Dubai",
            "question_date": "2026/07/01",
            "haystack_session_ids": ["noise", "answer"],
            "haystack_dates": ["2026/05/01", "2026/06/01"],
            "haystack_sessions": [
                [{"role": "user", "content": "Unrelated note."}],
                [
                    {
                        "role": "user",
                        "content": "I moved to Dubai.",
                        "has_answer": True,
                    }
                ],
            ],
        }
    ]

    runtime, _gold = longmemeval_candidates(source, tmp_path)
    files = sorted(Path(runtime[0].archive_path).glob("*.md"))

    assert [path.name for path in files] == ["session-0001.md"]


def test_longmemeval_evidence_points_to_turn_containing_answer(tmp_path: Path) -> None:
    source = [
        {
            "question_id": "q1",
            "question": "Which dance was mentioned?",
            "answer": "Hoop Dance",
            "question_date": "2026/07/01",
            "answer_session_ids": ["answer-session"],
            "haystack_session_ids": ["answer-session"],
            "haystack_dates": ["2026/06/01"],
            "haystack_sessions": [
                [
                    {
                        "role": "assistant",
                        "content": "The list included Hoop Dance.",
                    },
                    {
                        "role": "user",
                        "content": "What should I bring?",
                        "has_answer": True,
                    },
                ]
            ],
        }
    ]

    runtime, gold = longmemeval_candidates(source, tmp_path)

    session = Path(runtime[0].archive_path) / "session-0000.md"
    evidence_line = session.read_text(encoding="utf-8").splitlines()[
        gold[0].evidence[0].start_line - 1
    ]
    assert "Hoop Dance" in evidence_line
