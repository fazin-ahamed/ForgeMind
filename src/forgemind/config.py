from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    llama_server: Path
    model: Path
    host: str = "127.0.0.1"
    port: int = 8080
    context_tokens: int = 16_384
    batch_tokens: int = 512
    gpu_layers: int = -1
    max_output_tokens: int = 2_048
    timeout_seconds: float = 240.0

    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["llama_server"] = str(self.llama_server)
        data["model"] = str(self.model)
        return data

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> RuntimeConfig:
        server_text = env.get("FORGEMIND_LLAMA_SERVER")
        model_text = env.get("FORGEMIND_MODEL")
        if not server_text:
            raise ValueError("FORGEMIND_LLAMA_SERVER is required")
        if not model_text:
            raise ValueError("FORGEMIND_MODEL is required")
        host = env.get("FORGEMIND_HOST", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("ForgeMind must bind to a local host")
        server = Path(server_text).expanduser().resolve()
        model = Path(model_text).expanduser().resolve()
        if not server.is_file():
            raise ValueError(f"llama server not found: {server}")
        if not model.is_file():
            raise ValueError(f"model not found: {model}")
        return cls(
            llama_server=server,
            model=model,
            host=host,
            port=int(env.get("FORGEMIND_PORT", "8080")),
            context_tokens=min(int(env.get("FORGEMIND_CONTEXT", "16384")), 16_384),
        )
