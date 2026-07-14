from __future__ import annotations

import ctypes
import os
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import replace

import httpx

from forgemind.config import RuntimeConfig
from forgemind.domain import GenerationResult, HardwareProfile


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def parse_nvidia_smi(text: str, ram_mib: int) -> HardwareProfile:
    fields = [field.strip() for field in text.strip().split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"unexpected nvidia-smi output: {text!r}")
    return HardwareProfile(fields[0], int(fields[1].removesuffix(" MiB")), fields[2], ram_mib)


def parse_chat_response(payload: dict[str, object]) -> GenerationResult:
    choices = payload["choices"]
    usage = payload.get("usage", {})
    timings = payload.get("timings", {})
    message = choices[0]["message"]
    return GenerationResult(
        text=str(message["content"]),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        prompt_ms=float(timings.get("prompt_ms", 0.0)),
        generation_ms=float(timings.get("predicted_ms", 0.0)),
    )


def parse_tokenize_response(payload: dict[str, object]) -> int:
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("llama.cpp tokenize response omitted tokens")
    return len(tokens)


def parse_used_vram_mib(text: str) -> int:
    values = [int(line.strip()) for line in text.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory values")
    return max(values)


def used_vram_mib(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return parse_used_vram_mib(completed.stdout)


def physical_ram_mib() -> int:
    if os.name != "nt":
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1_048_576)
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.total_physical / 1_048_576)


def probe_hardware(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> HardwareProfile:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return parse_nvidia_smi(completed.stdout.splitlines()[0], physical_ram_mib())


class LlamaServer:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None

    def command(self) -> list[str]:
        return [
            str(self.config.llama_server),
            "-m",
            str(self.config.model),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "-c",
            str(self.config.context_tokens),
            "-b",
            str(self.config.batch_tokens),
            "-ngl",
            str(self.config.gpu_layers),
            "--metrics",
            "--cache-prompt",
        ]

    def start(self) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.wait_ready()

    def wait_ready(self, timeout_seconds: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                output = self.process.stdout.read()[-4_000:] if self.process.stdout else ""
                raise RuntimeError(
                    f"llama-server exited with {self.process.returncode}: {output}"
                )
            try:
                with urllib.request.urlopen(f"{self.config.server_url}/health", timeout=1) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.25)
        self.stop()
        raise TimeoutError("llama-server did not become healthy within 30 seconds")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


def start_with_single_fallback(
    config: RuntimeConfig,
    factory: Callable[[RuntimeConfig], LlamaServer] = LlamaServer,
) -> LlamaServer:
    first = factory(config)
    try:
        first.start()
        return first
    except RuntimeError as error:
        first.stop()
        if not any(word in str(error).lower() for word in ("out of memory", "cuda", "allocation")):
            raise
    fallback = replace(
        config,
        context_tokens=min(config.context_tokens, 8_192),
        batch_tokens=min(config.batch_tokens, 256),
    )
    second = factory(fallback)
    second.start()
    return second


class LlamaClient:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> GenerationResult:
        body: dict[str, object] = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens or self.config.max_output_tokens,
            "cache_prompt": True,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "forgemind_response",
                    "schema": json_schema,
                    "strict": True,
                },
            }
            body["chat_template_kwargs"] = {"enable_thinking": False}
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.post(
                f"{self.config.server_url}/v1/chat/completions",
                json=body,
            )
            response.raise_for_status()
            return parse_chat_response(response.json())

    def count_tokens(self, text: str) -> int:
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.post(
                f"{self.config.server_url}/tokenize", json={"content": text}
            )
            response.raise_for_status()
            return parse_tokenize_response(response.json())
