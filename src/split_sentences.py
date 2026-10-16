"""Split a passage into sentence."""

import re

MIN_CHARS = 40

# Split on whitespace that comes immediately after . ? or !
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(passage: str) -> list[str]:
    parts = [part.strip() for part in _BOUNDARY.split(passage)]
    kept = [part for part in parts if len(part) >= MIN_CHARS]
    if not kept:
        return [passage]
    return kept