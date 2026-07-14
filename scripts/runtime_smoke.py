from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from forgemind.config import RuntimeConfig
from forgemind.runtime import LlamaClient, probe_hardware, start_with_single_fallback


def main() -> int:
    requested = RuntimeConfig.from_env(os.environ)
    server = start_with_single_fallback(requested)
    prompt = [{"role": "user", "content": "Reply with exactly: ForgeMind ready /no_think"}]
    with server:
        client = LlamaClient(server.config)
        results = [client.complete(prompt, max_tokens=16) for _ in range(10)]
    if any("ForgeMind ready" not in result.text for result in results):
        raise SystemExit("smoke response mismatch")
    payload = {
        "hardware": asdict(probe_hardware()),
        "requested_runtime": requested.as_dict(),
        "effective_runtime": server.config.as_dict(),
        "runs": [asdict(result) | {"total_ms": result.total_ms} for result in results],
    }
    output = Path(".forgemind-private/results/m1-hardware-profile.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
