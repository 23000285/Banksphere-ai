"""Input validation and prompt-injection detection.

Checks for common prompt-injection patterns, role-assumption attempts,
and instruction-override attacks.  Returns structured results so callers
can decide whether to reject or flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("role_assumption", re.compile(
        r"(?:you are|act as|pretend to be|assume the role|roleplay as)\s",
        re.IGNORECASE,
    )),
    ("instruction_override", re.compile(
        r"(?:ignore (?:(?:all|previous|prior|above|your|my)\s+)*instructions|"
        r"disregard (?:(?:all|previous|prior|above|your|my)\s+)*(?:instructions|rules|guidelines)|"
        r"forget (?:(?:all|previous|prior|above|your|my)\s+)*(?:instructions|rules))",
        re.IGNORECASE,
    )),
    ("system_prompt_leak", re.compile(
        r"(?:reveal|show|print|output|display|repeat)\s+"
        r"(?:\w+\s+)*?(?:system|initial|original|hidden)\s*(?:prompt|instructions|message)",
        re.IGNORECASE,
    )),
    ("delimiter_injection", re.compile(
        r"(?:```|<\|im_start\||<\|im_end\||<system>|</system>|<\|endoftext\|>|<\|assistant\|>)",
        re.IGNORECASE,
    )),
    ("data_exfiltration", re.compile(
        r"(?:list all|dump all|show all|give me all|extract all)\s+"
        r"(?:customers?|users?|accounts?|records?|data|passwords?|secrets?|keys?)",
        re.IGNORECASE,
    )),
    ("sql_injection", re.compile(
        r"(?:'\s*(?:OR|AND)\s+['\d]|;\s*(?:DROP|DELETE|UPDATE|INSERT|SELECT)\s|"
        r"UNION\s+SELECT|--\s*$)",
        re.IGNORECASE,
    )),
]

MAX_QUERY_LENGTH = 2000


@dataclass
class InputCheckResult:
    is_safe: bool
    flagged_patterns: list[str] = field(default_factory=list)
    rejection_reason: str = ""


def check_input(query: str) -> InputCheckResult:
    """Check a user query for injection patterns and length limits.

    Returns an InputCheckResult indicating whether the input is safe.
    """
    if not query or not query.strip():
        return InputCheckResult(
            is_safe=False,
            rejection_reason="Query is empty.",
        )

    if len(query) > MAX_QUERY_LENGTH:
        return InputCheckResult(
            is_safe=False,
            rejection_reason=f"Query exceeds maximum length ({MAX_QUERY_LENGTH} characters).",
        )

    flagged: list[str] = []
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            flagged.append(name)

    if flagged:
        return InputCheckResult(
            is_safe=False,
            flagged_patterns=flagged,
            rejection_reason=f"Input flagged for: {', '.join(flagged)}.",
        )

    return InputCheckResult(is_safe=True)
