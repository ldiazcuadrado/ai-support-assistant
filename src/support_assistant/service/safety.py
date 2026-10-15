from __future__ import annotations

import re

from support_assistant.errors import InputRejectedError

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BLANK_RUNS = re.compile(r"\n{3,}")


def sanitize_input(text: str, max_chars: int) -> str:
    """Normalise a customer message and enforce the length limit."""
    cleaned = _BLANK_RUNS.sub("\n\n", _CONTROL.sub("", text.replace("\r\n", "\n"))).strip()
    if not cleaned:
        raise InputRejectedError("message is empty")
    if len(cleaned) > max_chars:
        raise InputRejectedError(f"message is longer than {max_chars} characters")
    return cleaned
