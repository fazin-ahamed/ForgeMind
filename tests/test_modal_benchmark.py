from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class _Image:
    @classmethod
    def from_registry(cls, *_args: object, **_kwargs: object) -> _Image:
        return cls()

    def __getattr__(self, _name: str):  # type: ignore[no-untyped-def]
        return lambda *_args, **_kwargs: self


class _Volume:
    @staticmethod
    def from_name(*_args: object, **_kwargs: object) -> object:
        return object()


class _App:
    def __init__(self, _name: str) -> None:
        pass

    def function(self, **_kwargs: object):  # type: ignore[no-untyped-def]
        def decorate(function):  # type: ignore[no-untyped-def]
            function.remote = function
            return function

        return decorate

    def local_entrypoint(self):  # type: ignore[no-untyped-def]
        return lambda function: function


def test_modal_entrypoint_forwards_adaptive_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modal = ModuleType("modal")
    modal.App = _App  # type: ignore[attr-defined]
    modal.Image = _Image  # type: ignore[attr-defined]
    modal.Volume = _Volume  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modal", modal)
    path = Path(__file__).parents[1] / "modal_benchmark.py"
    spec = importlib.util.spec_from_file_location("modal_benchmark_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs = iter(
        [SimpleNamespace(stdout="8ae2748\n"), SimpleNamespace(stdout="")]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: next(outputs))
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        module.evaluate, "remote", lambda *args: calls.append(args) or {}
    )

    module.main(run_group="fresh", systems="adaptive")

    assert calls == [("fresh", "8ae2748", False, "adaptive")]
