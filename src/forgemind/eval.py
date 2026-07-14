from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from forgemind.benchmark import (
    BenchmarkRun,
    CitationSpan,
    RuntimeCase,
    load_runtime_cases,
)
from forgemind.context import assemble_evidence
from forgemind.domain import (
    AnswerDraft,
    EvidenceItem,
    EvidencePack,
    GenerationResult,
    ReasoningLedger,
    SearchHit,
    VerifiedAnswer,
)
from forgemind.store import ForgeStore
from forgemind.verification import verify_answer


_SYSTEM_NAMES = ("raw", "vector", "hybrid", "forgemind", "raw32")


class EvaluationRetriever(Protocol):
    def search(self, query: str, limit: int = 20) -> list[SearchHit]: ...

    def search_vector(self, query: str, limit: int = 20) -> list[SearchHit]: ...


class EvaluationController(Protocol):
    def investigate(
        self, question: str, mode: str = "reason"
    ) -> tuple[
        AnswerDraft,
        ReasoningLedger,
        list[EvidencePack],
        list[GenerationResult],
    ]: ...


class EvaluationClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> GenerationResult: ...


def write_run(path: Path, run: BenchmarkRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(run.model_dump_json() + "\n")


def load_runs(path: Path) -> list[BenchmarkRun]:
    return [
        BenchmarkRun.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_cases(path: Path) -> list[RuntimeCase]:
    cases = load_runtime_cases(path)
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation cases contain duplicate IDs")
    return cases


def parse_system_names(text: str) -> list[str]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in _SYSTEM_NAMES]
    if unknown:
        raise ValueError(f"unknown evaluation systems: {', '.join(unknown)}")
    if len(names) != len(set(names)):
        raise ValueError("evaluation systems contain duplicates")
    if not names:
        raise ValueError("evaluation requires at least one system")
    return names


def freeze_results(
    directory: Path,
    cases: list[RuntimeCase],
    runs: list[BenchmarkRun],
    systems: list[str],
) -> None:
    run_path = directory / "runs.jsonl"
    summary_path = directory / "summary.json"
    if run_path.exists() or summary_path.exists():
        raise FileExistsError(f"frozen evaluation already exists: {directory}")
    expected = {(case.id, system) for case in cases for system in systems}
    actual = [(run.case_id, run.system) for run in runs]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("evaluation is missing or duplicates case/system runs")
    directory.mkdir(parents=True, exist_ok=True)
    for run in runs:
        write_run(run_path, run)
    summary_path.write_text(
        json.dumps(
            {
                "cases": len(cases),
                "systems": systems,
                "runs": len(runs),
                "errors": sum(run.error is not None for run in runs),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class EvaluationRunner:
    def __init__(
        self,
        systems: dict[str, Callable[[RuntimeCase], BenchmarkRun]],
        error_factory: Callable[[str, RuntimeCase, Exception], BenchmarkRun],
    ) -> None:
        self.systems = systems
        self.error_factory = error_factory

    def run(
        self,
        cases: list[RuntimeCase],
        order: list[str],
        existing: list[BenchmarkRun] | None = None,
        on_run: Callable[[BenchmarkRun], None] | None = None,
    ) -> list[BenchmarkRun]:
        completed = {
            (run.case_id, run.system) for run in existing or []
        }
        runs: list[BenchmarkRun] = []
        for case in sorted(cases, key=lambda item: item.id):
            for name in order:
                if (case.id, name) in completed:
                    continue
                try:
                    run = self.systems[name](case)
                    if run.system != name or run.case_id != case.id:
                        raise ValueError("system returned a mismatched run record")
                except Exception as error:
                    run = self.error_factory(name, case, error)
                runs.append(run)
                if on_run is not None:
                    on_run(run)
        return runs


def _citation(item: EvidenceItem) -> CitationSpan:
    return CitationSpan(
        source_id=item.source_id,
        source_sha256=item.source_sha256,
        path=item.path,
        start_line=item.start_line,
        end_line=item.end_line,
    )


def _unique_spans(items: list[EvidenceItem]) -> list[CitationSpan]:
    unique: dict[tuple[str, int, int], CitationSpan] = {}
    for item in items:
        span = _citation(item)
        unique.setdefault(
            (span.source_sha256, span.start_line, span.end_line), span
        )
    return list(unique.values())


class ControlledSystems:
    def __init__(
        self,
        store: ForgeStore,
        retriever: EvaluationRetriever,
        controller: EvaluationController,
        client: EvaluationClient,
        count_tokens: Callable[[str], int],
        vram_mib: Callable[[], int],
        run_group_id: str,
        model_sha256: str,
        config_sha256: str,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.controller = controller
        self.client = client
        self.count_tokens = count_tokens
        self.vram_mib = vram_mib
        self.run_group_id = run_group_id
        self.model_sha256 = model_sha256
        self.config_sha256 = config_sha256

    def vector(self, case: RuntimeCase) -> BenchmarkRun:
        return self._one_shot(
            "vector", case, self.retriever.search_vector(case.question, 20)
        )

    def raw(self, case: RuntimeCase) -> BenchmarkRun:
        return self._one_shot("raw", case, self.store.active_hits())

    def raw32(self, case: RuntimeCase) -> BenchmarkRun:
        return self._one_shot(
            "raw32",
            case,
            self.store.active_hits(),
            budget=32_000,
            prompt_limit=32_000,
        )

    def hybrid(self, case: RuntimeCase) -> BenchmarkRun:
        return self._one_shot(
            "hybrid", case, self.retriever.search(case.question, 20)
        )

    def forgemind(self, case: RuntimeCase) -> BenchmarkRun:
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        vram_before = self.vram_mib()
        draft, ledger, packs, generations = self.controller.investigate(
            case.question, "investigate"
        )
        answer = verify_answer(draft, ledger, packs, self.store)
        return self._record(
            "forgemind",
            case,
            draft,
            answer,
            packs,
            generations,
            started_at,
            started,
            vram_before,
        )

    def _one_shot(
        self,
        system: str,
        case: RuntimeCase,
        hits: list[SearchHit],
        budget: int = 12_000,
        prompt_limit: int | None = None,
    ) -> BenchmarkRun:
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        vram_before = self.vram_mib()
        pack = assemble_evidence(
            case.question,
            hits,
            self.count_tokens,
            budget=budget,
            archived_tokens=case.archived_tokens,
        )
        result = self.client.complete(
            [
                {
                    "role": "system",
                    "content": "Answer only from supplied evidence. Cite evidence IDs. /no_think",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": case.question,
                            "evidence": pack.model_payload(),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_schema=AnswerDraft.model_json_schema(),
        )
        draft = AnswerDraft.model_validate_json(result.text)
        ledger = ReasoningLedger(
            goal=case.question,
            cycle=1,
            retrieval_queries=[case.question],
            evidence_ids=[item.id for item in pack.items],
        )
        answer = verify_answer(draft, ledger, [pack], self.store)
        return self._record(
            system,
            case,
            draft,
            answer,
            [pack],
            [result],
            started_at,
            started,
            vram_before,
            prompt_limit=prompt_limit,
        )

    def _record(
        self,
        system: str,
        case: RuntimeCase,
        draft: AnswerDraft,
        answer: VerifiedAnswer,
        packs: list[EvidencePack],
        generations: list[GenerationResult],
        started_at: str,
        started: float,
        vram_before: int,
        prompt_limit: int | None = None,
    ) -> BenchmarkRun:
        valid_ids = {item.id for pack in packs for item in pack.items}
        attempted_ids = {
            item for claim in draft.claims for item in claim.evidence_ids
        }
        prompt_tokens = max(
            (item.prompt_tokens for item in generations), default=0
        )
        limit = prompt_limit if prompt_limit is not None else case.input_budget
        if prompt_tokens > limit:
            raise ValueError(
                f"prompt used {prompt_tokens} tokens; limit is {limit}"
            )
        run_id = hashlib.sha256(
            f"{self.run_group_id}\0{case.id}\0{system}".encode("utf-8")
        ).hexdigest()
        return BenchmarkRun(
            run_id=run_id,
            run_group_id=self.run_group_id,
            system=system,
            case_id=case.id,
            answer=None if answer.status == "abstained" else answer.summary,
            raw_outputs=[item.text for item in generations],
            citations=_unique_spans(answer.evidence),
            retrieved=_unique_spans(
                [item for pack in packs for item in pack.items]
            )[:20],
            retrieved_by_cycle=[_unique_spans(pack.items) for pack in packs],
            abstained=answer.status == "abstained",
            invalid_citations=len(attempted_ids - valid_ids),
            prompt_tokens=prompt_tokens,
            cumulative_prompt_tokens=sum(
                item.prompt_tokens for item in generations
            ),
            completion_tokens=sum(
                item.completion_tokens for item in generations
            ),
            retrieval_cycles=len(packs),
            latency_ms=(time.perf_counter() - started) * 1_000,
            peak_vram_mib=max(vram_before, self.vram_mib()),
            model_sha256=self.model_sha256,
            config_sha256=self.config_sha256,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    def error_record(
        self, system: str, case: RuntimeCase, error: Exception
    ) -> BenchmarkRun:
        now = datetime.now(timezone.utc).isoformat()
        run_id = hashlib.sha256(
            f"{self.run_group_id}\0{case.id}\0{system}".encode("utf-8")
        ).hexdigest()
        return BenchmarkRun(
            run_id=run_id,
            run_group_id=self.run_group_id,
            system=system,
            case_id=case.id,
            answer=None,
            raw_outputs=[],
            citations=[],
            retrieved=[],
            retrieved_by_cycle=[],
            abstained=True,
            invalid_citations=0,
            prompt_tokens=0,
            cumulative_prompt_tokens=0,
            completion_tokens=0,
            retrieval_cycles=0,
            latency_ms=0,
            peak_vram_mib=0,
            model_sha256=self.model_sha256,
            config_sha256=self.config_sha256,
            started_at=now,
            finished_at=now,
            error=str(error),
        )
