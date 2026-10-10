"""Deterministic mock LLM — returns canned responses without network or API keys.

Every response is clearly prefixed with [MOCK] so it is never mistaken for a
real LLM output.  The mock supports basic intent-matching so that agent routing
and tool-call flows can be developed and tested offline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

MOCK_PREFIX = "[MOCK] "

_INTENT_RESPONSES: list[tuple[str, str]] = [
    ("fraud", (
        "Transaction risk assessment: LOW risk (score 0.12). "
        "This is a simulated fraud-detection result."
    )),
    ("balance", (
        "The customer's current account balance is ₹45,230.50. "
        "This is a simulated response for development and testing."
    )),
    ("transaction", (
        "Here are the most recent transactions for the customer. "
        "All values are synthetic test data."
    )),
    ("credit", (
        "Estimated credit score: 720. Loan eligibility: pre-approved for ₹5,00,000. "
        "EMI at 10.5% for 36 months: ₹16,253. All values are synthetic."
    )),
    ("loan", (
        "Loan application summary: eligible based on synthetic credit profile. "
        "This is a mock response — not a real financial assessment."
    )),
    ("kyc", (
        "KYC policy: customers must provide government-issued photo ID and "
        "proof of address. Source: synthetic_policy_kyc_v1.md. "
        "This answer is based on simulated policy documents."
    )),
    ("compliance", (
        "Based on retrieved policy documents, the regulatory requirement is met. "
        "Source: synthetic_policy_compliance_v1.md. This is a simulated response."
    )),
    ("card", (
        "Card status: ACTIVE. A card-block request requires explicit human approval. "
        "This is a simulated response."
    )),
    ("transfer", (
        "Transfer request noted. This action requires human approval before execution. "
        "This is a simulated response."
    )),
]

_DEFAULT_RESPONSE = (
    "I've processed your request using the BankSphere AI system. "
    "This is a simulated response for development and testing purposes."
)


def _match_intent(prompt: str) -> str:
    """Match the prompt to a known intent keyword, case-insensitive."""
    lower = prompt.lower()
    for keyword, response in _INTENT_RESPONSES:
        if re.search(rf"\b{keyword}\b", lower):
            return response
    return _DEFAULT_RESPONSE


def _deterministic_id(prompt: str) -> str:
    """Produce a short stable ID from the prompt for tracing."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


@dataclass
class MockLLMResponse:
    content: str
    model: str = "mock-response"
    mock: bool = True
    trace_id: str = ""


@dataclass
class MockLLM:
    """Drop-in mock that returns deterministic, clearly-labelled responses."""

    model: str = "mock-response"
    _call_count: int = field(default=0, repr=False)

    def invoke(self, prompt: str) -> MockLLMResponse:
        self._call_count += 1
        raw = _match_intent(prompt)
        return MockLLMResponse(
            content=f"{MOCK_PREFIX}{raw}",
            model=self.model,
            mock=True,
            trace_id=_deterministic_id(prompt),
        )

    @property
    def call_count(self) -> int:
        return self._call_count
