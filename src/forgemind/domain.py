import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    gpu_name: str
    vram_mib: int
    driver_version: str
    ram_mib: int


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    prompt_ms: float
    generation_ms: float

    @property
    def total_ms(self) -> float:
        return self.prompt_ms + self.generation_ms


@dataclass(frozen=True, slots=True)
class SourceRecord:
    id: str
    path: str
    sha256: str
    modified_ns: int
    text: str

    @classmethod
    def from_text(cls, path: str, text: str, modified_ns: int) -> "SourceRecord":
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_id = hashlib.sha256(f"{path}\0{digest}".encode("utf-8")).hexdigest()
        return cls(source_id, path, digest, modified_ns, text)


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    id: str
    source_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectEvent:
    id: str
    commit: str
    occurred_at: str
    summary: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    score: float
    channels: tuple[str, ...]
