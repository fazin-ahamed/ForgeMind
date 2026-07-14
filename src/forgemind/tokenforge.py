from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


CANDIDATE = re.compile(
    r"(?:[A-Za-z]:)?[/\\][A-Za-z0-9_./\\-]{18,}"
    r"|[A-Za-z_][A-Za-z0-9_]{15,}"
    r"|https?://\S+"
)


@dataclass(frozen=True, slots=True)
class CompressedText:
    text: str
    aliases: dict[str, str]
    original_sha256: str


class TokenForge:
    def compress(self, text: str) -> CompressedText:
        counts: dict[str, int] = {}
        for match in CANDIDATE.finditer(text):
            value = match.group(0)
            counts[value] = counts.get(value, 0) + 1

        aliases: dict[str, str] = {}
        compressed = text
        index = 1
        for value, count in sorted(
            counts.items(), key=lambda pair: (-len(pair[0]) * pair[1], pair[0])
        ):
            if count < 2:
                continue
            alias = f"¤{index}"
            while alias in text or alias in aliases:
                index += 1
                alias = f"¤{index}"
            aliases[alias] = value
            compressed = compressed.replace(value, alias)
            index += 1

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return CompressedText(compressed, aliases, digest)

    def restore(self, compressed: CompressedText) -> str:
        text = compressed.text
        for alias, value in sorted(
            compressed.aliases.items(), key=lambda pair: -len(pair[0])
        ):
            text = text.replace(alias, value)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != compressed.original_sha256:
            raise ValueError("TokenForge round-trip checksum mismatch")
        return text
