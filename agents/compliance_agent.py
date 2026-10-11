"""Compliance agent — policy Q&A from retrieved evidence with citations.

Answers only from retrieved policy evidence.  When evidence is absent or
insufficient, the agent declines to answer rather than inventing content.
Calls only the 'retrieve_policy' tool via the central registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.base import AgentResult, BaseAgent
from gateway.audit import AuditTimer
from gateway.notifications import send_notification
from tools.compliance_tools import PolicyQueryRequest, PolicyQueryResponse
from tools.registry import ToolPermissionDenied, invoke_tool

AGENT_NAME = "compliance"

NO_EVIDENCE_RESPONSE = (
    "I could not find relevant policy evidence to answer this question. "
    "Please consult the compliance team directly for guidance."
)


@dataclass
class ComplianceAgent(BaseAgent):
    name: str = AGENT_NAME

    def answer_policy_question(
        self,
        query: str,
        customer_uid: str | None = None,
        notify: bool = False,
    ) -> AgentResult:
        """Retrieve policy evidence, synthesize an answer with citations.

        If *notify* is True a simulated notification stub event is logged
        (no real message is sent).
        """
        limit_hit = self._check_step_limit()
        if limit_hit:
            return limit_hit

        policy_resp = self._call_retrieve_policy(query)
        if policy_resp is None:
            return self._make_result("error", error=self._steps[-1].result_summary)

        if not policy_resp.has_evidence:
            self._record_step(
                "synthesize_answer", "success",
                "No relevant evidence found — declining to answer.",
            )
            result = self._make_result(
                "completed",
                response={
                    "answer": NO_EVIDENCE_RESPONSE,
                    "has_evidence": False,
                    "sources": [],
                    "query": query,
                },
            )
            if notify and customer_uid:
                self._send_notification(customer_uid, query, NO_EVIDENCE_RESPONSE)
            return result

        answer = self._synthesize_answer(query, policy_resp)
        sources = [s.model_dump(mode="json") for s in policy_resp.sources]

        self._record_step(
            "synthesize_answer", "success",
            f"Answer synthesized from {len(sources)} source(s).",
        )

        result = self._make_result(
            "completed",
            response={
                "answer": answer,
                "has_evidence": True,
                "sources": sources,
                "query": query,
            },
        )

        if notify and customer_uid:
            self._send_notification(customer_uid, query, answer)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_retrieve_policy(self, query: str) -> PolicyQueryResponse | None:
        with AuditTimer() as timer:
            try:
                resp = invoke_tool(
                    "retrieve_policy",
                    agent=AGENT_NAME,
                    request=PolicyQueryRequest(query=query),
                )
            except ToolPermissionDenied as exc:
                self._record_step(
                    "retrieve_policy", "denied", str(exc), timer.elapsed_ms,
                )
                return None
            except Exception as exc:
                self._record_step(
                    "retrieve_policy", "error", str(exc), timer.elapsed_ms,
                )
                return None

        self._record_step(
            "retrieve_policy", "success",
            f"Retrieved {resp.total_results} chunk(s) for query.",
            timer.elapsed_ms,
        )
        return resp

    def _synthesize_answer(
        self,
        query: str,
        policy_resp: PolicyQueryResponse,
    ) -> str:
        """Build an evidence-grounded answer with source citations.

        This is a deterministic, template-based synthesis (no LLM call)
        suitable for the mock-first MVP.  A future version could pass the
        evidence to an LLM for a more natural response.
        """
        parts: list[str] = [f"Based on retrieved policy evidence for: \"{query}\"\n"]

        seen_sources: set[str] = set()
        for hit in policy_resp.results:
            source = hit["source"]
            heading = hit["heading"]
            text = hit["text"]
            citation = f"[{source} — {heading}]"

            excerpt = text[:500] + ("..." if len(text) > 500 else "")
            parts.append(f"{citation}\n{excerpt}\n")
            seen_sources.add(source)

        if seen_sources:
            parts.append(
                "Sources consulted: " + ", ".join(sorted(seen_sources))
            )

        return "\n".join(parts)

    def _send_notification(
        self,
        customer_uid: str,
        query: str,
        answer: str,
    ) -> None:
        with AuditTimer() as timer:
            event = send_notification(
                channel="in_app",
                recipient=customer_uid,
                subject=f"Policy inquiry: {query[:80]}",
                body=answer[:1000],
                event_type="policy_response",
                trace_id=self.trace_id,
                metadata={"agent": AGENT_NAME, "query": query},
            )
        self._record_step(
            "send_notification", "success",
            f"Simulated {event.channel} notification logged.",
            timer.elapsed_ms,
        )
