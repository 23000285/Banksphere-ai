"""Practical PII masking for log output and responses.

Masks common Indian PII patterns: Aadhaar, PAN, phone numbers, email
addresses, and account-like digit sequences.  This is a best-effort
regex approach suitable for an MVP — it will not catch every format.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("aadhaar", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "XXXX-XXXX-XXXX"),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "XXXXX0000X"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "***@***.***"),
    ("phone", re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"), "XXXXXXXXXX"),
    ("account_number", re.compile(r"\b\d{9,18}\b"), "ACCT-****"),
]


def mask_pii(text: str) -> str:
    """Replace recognised PII patterns with safe placeholders."""
    result = text
    for _name, pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def contains_pii(text: str) -> list[str]:
    """Return a list of PII category names found in the text."""
    found: list[str] = []
    for name, pattern, _repl in _PATTERNS:
        if pattern.search(text):
            found.append(name)
    return found
