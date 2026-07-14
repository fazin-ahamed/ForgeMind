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
