"""Load and chunk policy documents from the corpus directory."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

POLICIES_DIR = Path(__file__).parent / "policies"

CATEGORY_MAP: dict[str, str] = {
    "kyc_verification": "KYC",
    "minor_accounts": "Minor Accounts",
    "loan_eligibility": "Loans",
    "fraud_disputes": "Fraud",
    "card_rules": "Cards",
    "account_closure": "Accounts",
}


@dataclass
class PolicyChunk:
    chunk_id: str
    source: str
    category: str
    heading: str
    text: str
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)


def _extract_category(text: str, filename: str) -> str:
    """Extract category from document metadata or fall back to filename map."""
    match = re.search(r"\*\*Category:\*\*\s*(.+)", text)
    if match:
        return match.group(1).strip()
    stem = Path(filename).stem
    return CATEGORY_MAP.get(stem, "General")


def _split_by_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, body) pairs by ## headings."""
    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_heading, body))
            current_heading = line.lstrip("# ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    return sections


def load_policy_documents(
    policies_dir: Path | None = None,
) -> list[PolicyChunk]:
    """Load all .md files from the policies directory and split into chunks.

    Each ## section becomes one chunk, preserving source and category metadata.
    """
    directory = policies_dir or POLICIES_DIR
    if not directory.exists():
        return []

    chunks: list[PolicyChunk] = []
    md_files = sorted(directory.glob("*.md"))

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        source = md_file.name
        category = _extract_category(text, source)
        sections = _split_by_sections(text)

        for idx, (heading, body) in enumerate(sections):
            chunk_id = f"{md_file.stem}::chunk_{idx}"
            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    source=source,
                    category=category,
                    heading=heading,
                    text=body,
                    chunk_index=idx,
                    metadata={
                        "source": source,
                        "category": category,
                        "heading": heading,
                    },
                )
            )

    return chunks
