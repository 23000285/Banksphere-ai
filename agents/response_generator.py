"""LLM-powered response generation from verified agent results.

Takes structured tool/model outputs and produces a natural-language narrative.
The LLM explains results but never invents data, overrides scores, or
bypasses authorization. Falls back gracefully when the provider is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from gateway.llm_config import LLMConfig, load_llm_config
from gateway.llm_provider import get_llm
from gateway.pii import mask_pii

logger = logging.getLogger("banksphere.response_generator")

_SYSTEM_PREAMBLE = (
    "You are a helpful banking assistant for BankSphere AI. "
    "You explain verified banking data to customers in clear, friendly language. "
    "Rules:\n"
    "- Only reference data provided below. Never invent balances, scores, dates, "
    "account numbers, transactions, policy details, or citations.\n"
    "- Do not override fraud scores, credit decisions, or eligibility rules.\n"
    "- Do not authorize or execute any action — only explain results.\n"
    "- Keep responses concise (2-4 sentences for simple queries, up to a short "
    "paragraph for complex ones).\n"
    "- Use Indian Rupee (₹) formatting for amounts.\n"
    "- All data is synthetic and for demonstration only.\n"
)


def _build_banking_prompt(query: str, data: dict) -> str:
    accounts = data.get("accounts", [])
    transactions = data.get("transactions", [])
    parts = [f"Customer query: {query}\n\nVerified data:\n"]
    for a in accounts:
        parts.append(
            f"- Account {a.get('account_number')}: {a.get('account_type')}, "
            f"balance ₹{a.get('balance')}, status {a.get('status')}"
        )
    if transactions:
        parts.append(f"\nRecent transactions ({len(transactions)}):")
        for t in transactions[:5]:
            parts.append(
                f"- {t.get('timestamp', '')[:10]}: {t.get('txn_type')} ₹{t.get('amount')} "
                f"at {t.get('merchant')} via {t.get('channel')}"
            )
    parts.append("\nProvide a brief, friendly summary of this account information.")
    return "\n".join(parts)


def _build_fraud_prompt(query: str, data: dict) -> str:
    scored = data.get("scored_transactions", [])
    parts = [f"Customer query: {query}\n\nVerified fraud analysis:\n"]
    for s in scored:
        parts.append(
            f"- Transaction {s.get('transaction_uid', '')[:8]}...: "
            f"₹{s.get('amount')}, fraud_score={s.get('fraud_score')}, "
            f"risk={s.get('risk_category')}, recommendation={s.get('recommendation')}"
        )
        if s.get("top_features"):
            feats = ", ".join(
                f"{list(f.keys())[0]}={list(f.values())[0]:.3f}"
                for f in s["top_features"][:3]
                if isinstance(f, dict)
            )
            parts.append(f"  Key factors: {feats}")
    parts.append(
        "\nExplain the fraud analysis results. "
        "Do not change scores or recommendations. "
        "If any transaction is high-risk, emphasize caution."
    )
    return "\n".join(parts)


def _build_credit_prompt(query: str, data: dict) -> str:
    parts = [f"Customer query: {query}\n\nVerified credit assessment:\n"]
    parts.append(f"- Eligible: {data.get('eligible')}")
    parts.append(f"- Risk category: {data.get('risk_category')}")
    parts.append(f"- Default probability: {data.get('default_probability')}")
    parts.append(f"- Monthly EMI: ₹{data.get('emi')}")
    if data.get("max_eligible_amount"):
        parts.append(f"- Max eligible amount: ₹{data.get('max_eligible_amount')}")
    if data.get("debt_to_income_ratio") is not None:
        parts.append(f"- Debt-to-income ratio: {data.get('debt_to_income_ratio')}")
    reasons = data.get("reason_codes", [])
    if reasons:
        parts.append(f"- Reason codes: {', '.join(reasons)}")
    parts.append(
        "\nExplain this credit assessment in plain language. "
        "Do not alter the eligibility decision, EMI, or amounts."
    )
    return "\n".join(parts)


def _build_compliance_prompt(query: str, data: dict) -> str:
    answer = data.get("answer", "")
    sources = data.get("sources", [])
    if not answer:
        return ""
    source_names = [s.get("source", "") for s in sources if s.get("source")]
    parts = [
        f"Customer query: {query}\n\n"
        f"Retrieved policy evidence:\n{answer}\n\n"
        f"Sources: {', '.join(source_names) if source_names else 'none'}\n\n"
        "Summarize the policy answer concisely. "
        "Preserve all source citations exactly as given. "
        "Do not add policy requirements not present in the evidence. "
        "If evidence is insufficient, say so."
    ]
    return "\n".join(parts)


def _build_card_block_prompt(query: str, data: dict) -> str:
    parts = [f"Customer query: {query}\n\nAction result:\n"]
    parts.append(f"- Action: {data.get('action')}")
    parts.append(f"- Account: {data.get('account_number')}")
    parts.append(f"- Status: {data.get('status')}")
    parts.append(f"- Message: {data.get('message')}")
    if data.get("requires_approval"):
        parts.append("- This action requires human approval before execution.")
    parts.append(
        "\nExplain the card block status briefly. "
        "Do not claim the card is blocked if status is pending_approval."
    )
    return "\n".join(parts)


_PROMPT_BUILDERS = {
    "banking": _build_banking_prompt,
    "fraud": _build_fraud_prompt,
    "credit": _build_credit_prompt,
    "compliance": _build_compliance_prompt,
    "card_block": _build_card_block_prompt,
}


def generate_narrative(
    intent: str,
    query: str,
    response_data: dict[str, Any],
    config: LLMConfig | None = None,
) -> str:
    """Generate a natural-language narrative from verified agent results.

    Returns empty string on failure or unsupported intent — callers should
    fall back to the existing structured/template display.
    """
    if config is None:
        config = load_llm_config()

    builder = _PROMPT_BUILDERS.get(intent)
    if builder is None:
        return ""

    try:
        prompt_body = builder(query, response_data)
    except Exception:
        logger.warning("Failed to build prompt for intent=%s", intent, exc_info=True)
        return ""

    if not prompt_body:
        return ""

    full_prompt = f"{_SYSTEM_PREAMBLE}\n{mask_pii(prompt_body)}"

    try:
        llm = get_llm(config)
        response = llm.invoke(full_prompt)
        narrative = response.content or ""
        return mask_pii(narrative)
    except Exception:
        logger.warning("LLM call failed for intent=%s, falling back", intent, exc_info=True)
        return ""
