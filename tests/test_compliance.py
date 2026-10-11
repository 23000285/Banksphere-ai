"""Tests for Phase 9: Compliance agent, notification stub, and supervisor routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.loader import PolicyChunk, load_policy_documents
from rag.chroma_store import ingest_chunks
from tools.compliance_tools import (
    PolicyQueryRequest,
    PolicyQueryResponse,
    init_retriever,
    retrieve_policy,
)
from tools.registry import (
    ToolPermissionDenied,
    invoke_tool,
    register_compliance_tools,
    register_tool,
)
from agents.compliance_agent import ComplianceAgent, NO_EVIDENCE_RESPONSE
from agents.base import AgentResult
from gateway.notifications import (
    NotificationEvent,
    clear_event_log,
    get_event_log,
    send_notification,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_notification_log():
    """Clear the notification event log before each test."""
    clear_event_log()
    yield
    clear_event_log()


@pytest.fixture()
def policy_chunks() -> list[PolicyChunk]:
    return load_policy_documents()


@pytest.fixture()
def chroma_dir(tmp_path: Path) -> Path:
    d = tmp_path / "chroma_compliance"
    d.mkdir()
    return d


@pytest.fixture()
def retriever_ready(policy_chunks: list[PolicyChunk], chroma_dir: Path):
    """Ingest the corpus, init the retriever, and register the tool."""
    ingest_chunks(policy_chunks, persist_dir=chroma_dir)
    init_retriever(chroma_persist_dir=chroma_dir)
    register_compliance_tools()
    yield
    import tools.compliance_tools as ct
    ct._retriever = None


# ---------------------------------------------------------------------------
# Notification Stub
# ---------------------------------------------------------------------------


class TestNotificationStub:
    def test_send_returns_event(self) -> None:
        event = send_notification(
            channel="email",
            recipient="CUST001",
            subject="Test subject",
            body="Test body",
            event_type="info",
        )
        assert isinstance(event, NotificationEvent)
        assert event.channel == "email"
        assert event.recipient == "CUST001"
        assert event.delivered is False

    def test_event_logged(self) -> None:
        send_notification(
            channel="sms", recipient="CUST002",
            subject="Alert", body="Body", event_type="alert",
        )
        log = get_event_log()
        assert len(log) == 1
        assert log[0].channel == "sms"
        assert log[0].recipient == "CUST002"

    def test_multiple_events(self) -> None:
        for i in range(3):
            send_notification(
                channel="push", recipient=f"CUST{i:03d}",
                subject=f"Subject {i}", body=f"Body {i}",
            )
        assert len(get_event_log()) == 3

    def test_clear_log(self) -> None:
        send_notification(
            channel="in_app", recipient="CUST001",
            subject="S", body="B",
        )
        assert len(get_event_log()) == 1
        clear_event_log()
        assert len(get_event_log()) == 0

    def test_event_has_trace_id(self) -> None:
        event = send_notification(
            channel="email", recipient="CUST001",
            subject="S", body="B", trace_id="abc123",
        )
        assert event.trace_id == "abc123"

    def test_event_has_timestamp(self) -> None:
        event = send_notification(
            channel="email", recipient="CUST001",
            subject="S", body="B",
        )
        assert event.timestamp

    def test_event_log_returns_dict(self) -> None:
        event = send_notification(
            channel="email", recipient="CUST001",
            subject="S", body="B", event_type="info",
        )
        record = event.log()
        assert record["simulated"] is True
        assert record["delivered"] is False
        assert record["channel"] == "email"

    def test_metadata_preserved(self) -> None:
        event = send_notification(
            channel="email", recipient="CUST001",
            subject="S", body="B",
            metadata={"agent": "compliance", "extra": "data"},
        )
        assert event.metadata["agent"] == "compliance"
        assert event.metadata["extra"] == "data"


# ---------------------------------------------------------------------------
# Compliance Tool — retrieve_policy
# ---------------------------------------------------------------------------


class TestRetrievePolicyTool:
    def test_returns_results_for_kyc_query(self, retriever_ready) -> None:
        resp = retrieve_policy(PolicyQueryRequest(query="KYC verification"))
        assert isinstance(resp, PolicyQueryResponse)
        assert resp.has_evidence is True
        assert resp.total_results > 0
        assert any("kyc_verification.md" == s.source for s in resp.sources)

    def test_returns_results_for_loan_query(self, retriever_ready) -> None:
        resp = retrieve_policy(PolicyQueryRequest(query="loan eligibility requirements"))
        assert resp.has_evidence is True
        assert any("loan_eligibility.md" == s.source for s in resp.sources)

    def test_sources_have_metadata(self, retriever_ready) -> None:
        resp = retrieve_policy(PolicyQueryRequest(query="card blocking rules"))
        for s in resp.sources:
            assert s.chunk_id
            assert s.source
            assert s.category
            assert s.rrf_score > 0

    def test_top_k_limits_results(self, retriever_ready) -> None:
        resp = retrieve_policy(PolicyQueryRequest(query="banking policy", top_k=2))
        assert resp.total_results <= 2

    def test_uninitialized_retriever_returns_empty(self) -> None:
        import tools.compliance_tools as ct
        saved = ct._retriever
        ct._retriever = None
        try:
            resp = retrieve_policy(PolicyQueryRequest(query="anything"))
            assert resp.has_evidence is False
            assert resp.total_results == 0
        finally:
            ct._retriever = saved

    def test_invoke_via_registry_compliance_agent(self, retriever_ready) -> None:
        resp = invoke_tool(
            "retrieve_policy",
            agent="compliance",
            request=PolicyQueryRequest(query="KYC"),
        )
        assert isinstance(resp, PolicyQueryResponse)
        assert resp.has_evidence is True

    def test_invoke_via_registry_denied_for_banking(self, retriever_ready) -> None:
        with pytest.raises(ToolPermissionDenied):
            invoke_tool(
                "retrieve_policy",
                agent="banking",
                request=PolicyQueryRequest(query="KYC"),
            )


# ---------------------------------------------------------------------------
# Compliance Agent
# ---------------------------------------------------------------------------


class TestComplianceAgent:
    def test_answers_kyc_question(self, retriever_ready) -> None:
        agent = ComplianceAgent()
        result = agent.answer_policy_question("What documents are needed for KYC?")
        assert result.status == "completed"
        assert result.response["has_evidence"] is True
        assert "kyc_verification.md" in result.response["answer"]

    def test_answers_loan_question(self, retriever_ready) -> None:
        agent = ComplianceAgent()
        result = agent.answer_policy_question("What is the minimum income for a personal loan?")
        assert result.status == "completed"
        assert result.response["has_evidence"] is True
        assert any(
            "loan_eligibility.md" == s["source"]
            for s in result.response["sources"]
        )

    def test_answers_card_question(self, retriever_ready) -> None:
        agent = ComplianceAgent()
        result = agent.answer_policy_question("How do I block my card?")
        assert result.status == "completed"
        assert result.response["has_evidence"] is True

    def test_cites_sources(self, retriever_ready) -> None:
        agent = ComplianceAgent()
        result = agent.answer_policy_question("What is the KYC renewal period?")
        assert result.response["sources"]
        for s in result.response["sources"]:
            assert s["source"].endswith(".md")
            assert s["category"]
            assert s["chunk_id"]

    def test_sources_consulted_in_answer(self, retriever_ready) -> None:
        agent = ComplianceAgent()
        result = agent.answer_policy_question("What are the KYC requirements?")
        assert "Sources consulted:" in result.response["answer"]

    def test_declines_unsupported_question(self, retriever_ready) -> None:
        """Agent should return no-evidence response for gibberish queries.

        Note: ChromaDB may still return distance-based results for nonsense,
        so we test with a truly unrelated topic where BM25 finds nothing
        and the retriever is uninitialized.
        """
        import tools.compliance_tools as ct
        saved = ct._retriever
        ct._retriever = None
        try:
            agent = ComplianceAgent()
            register_compliance_tools()
            result = agent.answer_policy_question("xyzzyplugh")
            assert result.status == "completed"
            assert result.response["has_evidence"] is False
            assert result.response["answer"] == NO_EVIDENCE_RESPONSE
        finally:
            ct._retriever = saved

    def test_records_steps(self, retriever_ready) -> None:
        agent = ComplianceAgent()
        result = agent.answer_policy_question("What is KYC?")
        assert result.steps
        tool_names = [s.tool_name for s in result.steps]
        assert "retrieve_policy" in tool_names

    def test_agent_name_and_trace_id(self, retriever_ready) -> None:
        agent = ComplianceAgent()
        result = agent.answer_policy_question("loan eligibility")
        assert result.agent == "compliance"
        assert result.trace_id

    def test_step_limit_enforced(self, retriever_ready) -> None:
        agent = ComplianceAgent(max_steps=0)
        result = agent.answer_policy_question("anything")
        assert result.status == "max_steps_reached"

    def test_notification_sent_when_requested(self, retriever_ready) -> None:
        agent = ComplianceAgent()
        result = agent.answer_policy_question(
            "KYC requirements",
            customer_uid="CUST001",
            notify=True,
        )
        assert result.status == "completed"
        log = get_event_log()
        assert len(log) == 1
        assert log[0].channel == "in_app"
        assert log[0].recipient == "CUST001"
        assert log[0].event_type == "policy_response"
        assert log[0].delivered is False

    def test_no_notification_when_not_requested(self, retriever_ready) -> None:
        agent = ComplianceAgent()
        agent.answer_policy_question("KYC requirements")
        assert len(get_event_log()) == 0

    def test_notification_on_no_evidence(self, retriever_ready) -> None:
        import tools.compliance_tools as ct
        saved = ct._retriever
        ct._retriever = None
        try:
            agent = ComplianceAgent()
            register_compliance_tools()
            result = agent.answer_policy_question(
                "xyzzyplugh",
                customer_uid="CUST001",
                notify=True,
            )
            assert result.response["has_evidence"] is False
            log = get_event_log()
            assert len(log) == 1
            assert NO_EVIDENCE_RESPONSE in log[0].body
        finally:
            ct._retriever = saved

    def test_initialized_retriever_empty_corpus(self, tmp_path: Path) -> None:
        """An initialized retriever over an empty corpus returns no evidence.

        This exercises the code path where _retriever is not None but the
        query returns zero results — distinct from the _retriever=None path.
        """
        import tools.compliance_tools as ct

        empty_policies = tmp_path / "empty_policies"
        empty_policies.mkdir()
        empty_chroma = tmp_path / "empty_chroma"
        empty_chroma.mkdir()

        saved = ct._retriever
        try:
            init_retriever(
                chroma_persist_dir=empty_chroma,
                policies_dir=empty_policies,
            )
            register_compliance_tools()

            resp = retrieve_policy(PolicyQueryRequest(query="KYC verification"))
            assert resp.has_evidence is False
            assert resp.total_results == 0

            agent = ComplianceAgent()
            result = agent.answer_policy_question("KYC verification")
            assert result.status == "completed"
            assert result.response["has_evidence"] is False
            assert result.response["answer"] == NO_EVIDENCE_RESPONSE
            assert result.response["sources"] == []
        finally:
            ct._retriever = saved


# ---------------------------------------------------------------------------
# Supervisor Routing — compliance intent
# ---------------------------------------------------------------------------


class TestSupervisorComplianceRouting:
    def test_classify_kyc_as_compliance(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("What is the KYC policy?") == "compliance"

    def test_classify_policy_as_compliance(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("Tell me about the card policy") == "compliance"

    def test_classify_compliance_keyword(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("compliance requirements for new accounts") == "compliance"

    def test_classify_minor_account_as_compliance(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("What are minor account rules?") == "compliance"

    def test_classify_guardian_as_compliance(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("guardian requirements for opening account") == "compliance"

    def test_classify_aml_as_compliance(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("What are the AML regulations?") == "compliance"

    def test_classify_dormant_as_compliance(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("When does an account become dormant?") == "compliance"

    def test_fraud_still_routes_correctly(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("Is this transaction fraudulent?") == "fraud"

    def test_banking_still_routes_correctly(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("Show my account balance") == "banking"

    def test_credit_still_routes_correctly(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("Am I eligible for a loan?") == "credit"

    def test_unsupported_still_routes_correctly(self) -> None:
        from agents.supervisor import classify_intent
        assert classify_intent("What is the weather today?") == "unsupported"
