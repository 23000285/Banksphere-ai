"""BankSphere AI — Streamlit banking chat interface.

Run: streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)

import streamlit as st

st.set_page_config(page_title="BankSphere AI", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------------------
# One-time initialisation (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def _init_backend():
    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker
    from ml.db_engine import init_db
    from ml.seed_data import seed_database
    from ml.fraud_model import train_fraud_models
    from ml.credit_model import train_credit_model
    from ml.db_models import Customer
    from tools.registry import (
        register_banking_tools,
        register_fraud_tools,
        register_credit_tools,
        register_compliance_tools,
        _TOOL_REGISTRY,
    )

    db_url = "sqlite:///data/banksphere.db"
    engine = init_db(db_url)
    seed_database(db_url)

    fraud_artifacts = train_fraud_models()
    credit_artifacts = train_credit_model()

    _TOOL_REGISTRY.clear()
    register_banking_tools()
    register_fraud_tools()
    register_credit_tools()
    register_compliance_tools()

    Session = sessionmaker(bind=engine)
    session = Session()
    customers = session.execute(
        select(Customer).order_by(Customer.first_name).limit(50)
    ).scalars().all()
    customer_map = {
        f"{c.first_name} {c.last_name} ({c.customer_uid[:8]}...)": c.customer_uid
        for c in customers
    }
    session.close()

    return engine, fraud_artifacts, credit_artifacts, customer_map


# ---------------------------------------------------------------------------
# Response formatting helpers
# ---------------------------------------------------------------------------

def _format_banking(resp: dict) -> str:
    lines = ["**Account Summary**\n"]
    for acct in resp.get("accounts", []):
        status_icon = "🟢" if acct.get("status") == "active" else "🔴"
        lines.append(
            f"- {status_icon} **{acct.get('account_type', 'Account')}** "
            f"`{acct.get('account_number', 'N/A')}` — "
            f"₹{float(acct.get('balance', 0)):,.2f} ({acct.get('currency', 'INR')})"
        )
    txns = resp.get("transactions", [])
    if txns:
        lines.append(f"\n**Recent Transactions** ({len(txns)} shown)\n")
        lines.append("| Date | Type | Category | Amount | Merchant |")
        lines.append("|---|---|---|---|---|")
        for t in txns[:10]:
            ts = str(t.get("timestamp", ""))[:10]
            flag = " 🚩" if t.get("is_flagged") else ""
            lines.append(
                f"| {ts} | {t.get('txn_type', '')} | {t.get('category', '')} "
                f"| ₹{float(t.get('amount', 0)):,.2f}{flag} | {t.get('merchant', '')} |"
            )
    return "\n".join(lines)


def _format_fraud(resp: dict) -> str:
    scored = resp.get("scored_transactions", [])
    if not scored:
        return "No transactions scored."
    lines = [f"**Fraud Analysis** — {len(scored)} transactions scored\n"]
    for txn in scored:
        score = float(txn.get("fraud_score", 0))
        risk = txn.get("risk_category", "unknown")
        rec = txn.get("recommendation", "")
        icon = {"allow": "🟢", "verify": "🟡", "block": "🔴"}.get(rec, "⚪")
        lines.append(
            f"- {icon} `{txn.get('transaction_uid', '')[:8]}...` — "
            f"₹{float(txn.get('amount', 0)):,.2f} | "
            f"Score: {score:.4f} | Risk: **{risk}** | Rec: **{rec}**"
        )
        if txn.get("requires_approval"):
            lines.append("  - ⚠️ *Requires human approval before action*")
    return "\n".join(lines)


def _format_credit(resp: dict) -> str:
    lines = ["**Loan Eligibility Assessment**\n"]
    elig = resp.get("eligible")
    if elig is not None:
        icon = "✅" if elig else "❌"
        lines.append(f"- {icon} **Eligible:** {'Yes' if elig else 'No'}")
    for key in ("risk_category", "default_probability", "monthly_emi",
                "total_payment", "total_interest"):
        val = resp.get(key)
        if val is not None:
            label = key.replace("_", " ").title()
            if isinstance(val, float) and val < 1:
                lines.append(f"- **{label}:** {val:.4f}")
            elif isinstance(val, (int, float)):
                lines.append(f"- **{label}:** ₹{val:,.2f}")
            else:
                lines.append(f"- **{label}:** {val}")
    reasons = resp.get("reason_codes") or resp.get("reasons")
    if reasons:
        lines.append("\n**Reason Codes:**")
        for r in reasons:
            lines.append(f"- {r}")
    return "\n".join(lines)


def _format_compliance(resp: dict) -> str:
    answer = resp.get("answer", "")
    if answer:
        return answer
    return "*No policy information available for this query.*"


def _format_response(intent: str, response: dict, trace_id: str) -> str:
    if intent == "banking":
        return _format_banking(response)
    elif intent == "fraud":
        return _format_fraud(response)
    elif intent == "credit":
        return _format_credit(response)
    elif intent == "compliance":
        return _format_compliance(response)
    return f"*Response received (trace: {trace_id[:8]})*"


def _render_trace(result_dict: dict, result_data: dict | None = None):
    """Render expandable agent/tool trace and optional visualizations."""
    agent_name = result_dict.get("agent", "unknown")
    trace_id = result_dict.get("trace_id", "")
    status = result_dict.get("status", "")
    steps = result_dict.get("steps", [])

    with st.expander(f"Agent trace — {agent_name} ({status}) [{trace_id[:8]}]"):
        if steps:
            for s in steps:
                tool = s.get("tool_name", "?")
                outcome = s.get("outcome", "?")
                dur = s.get("duration_ms", 0)
                icon = "✅" if outcome == "success" else "❌"
                st.markdown(f"{icon} **{tool}** — {outcome} ({dur:.1f}ms)")
                summary = s.get("result_summary", "")
                if summary:
                    st.caption(summary)
        else:
            st.caption("No tool steps recorded.")

    if result_data and result_data.get("scored_transactions"):
        _render_risk_chart(result_data)


def _render_risk_chart(result_data: dict):
    """Render risk-score bar chart for fraud results."""
    import pandas as pd

    scored = result_data["scored_transactions"]
    rows = []
    for txn in scored:
        rows.append({
            "Transaction": txn.get("transaction_uid", "")[:8] + "...",
            "Amount": float(txn.get("amount", 0)),
            "Fraud Score": float(txn.get("fraud_score", 0)),
            "Risk": txn.get("risk_category", "unknown"),
            "Recommendation": txn.get("recommendation", ""),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    st.subheader("Transaction Risk Scores")
    st.bar_chart(df.set_index("Transaction")["Fraud Score"], color="#ff4b4b")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Initialise backend
# ---------------------------------------------------------------------------

engine, fraud_artifacts, credit_artifacts, customer_map = _init_backend()


# ---------------------------------------------------------------------------
# Sidebar — customer selector and approval panel
# ---------------------------------------------------------------------------

st.sidebar.title("BankSphere AI")
st.sidebar.markdown("Multi-agent banking assistant")

selected_label = st.sidebar.selectbox("Customer", list(customer_map.keys()))
customer_uid = customer_map[selected_label]
st.sidebar.caption(f"UID: `{customer_uid}`")

st.sidebar.divider()
st.sidebar.subheader("Pending Approvals")

from gateway.approval import get_pending_approvals, resolve_approval

pending = get_pending_approvals(customer_uid=customer_uid)
if not pending:
    st.sidebar.info("No pending approvals.")
else:
    for req in pending:
        with st.sidebar.container(border=True):
            st.sidebar.markdown(
                f"**{req.action}** — `{req.request_id[:8]}...`"
            )
            if req.reason:
                st.sidebar.caption(req.reason)
            col_a, col_r = st.sidebar.columns(2)
            if col_a.button("Approve", key=f"approve_{req.request_id}"):
                resolve_approval(
                    req.request_id, approved=True, resolved_by="streamlit_user",
                )
                st.rerun()
            if col_r.button("Reject", key=f"reject_{req.request_id}"):
                resolve_approval(
                    req.request_id, approved=False, resolved_by="streamlit_user",
                )
                st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    "**Sample queries:**\n"
    "- Show my balance\n"
    "- Is this transaction suspicious?\n"
    "- Am I eligible for a loan?\n"
    "- What is the KYC policy?\n"
    "- Block my card\n"
)


# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------

st.title("Banking Agent Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            _render_trace(msg["trace"], msg.get("result_data"))


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about your account, transactions, loans, or policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            from sqlalchemy.orm import sessionmaker
            from agents.supervisor import RoutingRequest, run_supervisor
            from gateway.pii import mask_pii

            Session = sessionmaker(bind=engine)
            session = Session()
            try:
                result = run_supervisor(
                    RoutingRequest(query=prompt, customer_uid=customer_uid),
                    session=session,
                    fraud_artifacts=fraud_artifacts,
                    credit_artifacts=credit_artifacts,
                )
            finally:
                session.close()

        result_dict = result.result
        response_data = (
            result_dict.get("response", {})
            if isinstance(result_dict, dict)
            else {}
        )

        if result.status == "rejected":
            reply = f"**Request rejected:** {result.error}"
        elif result.error:
            reply = f"**Error:** {mask_pii(result.error)}"
        elif result.intent == "unsupported":
            reply = (
                "I'm not sure how to help with that. I can assist with "
                "banking, fraud checks, loan eligibility, and policy questions."
            )
        else:
            reply = _format_response(result.intent, response_data, result.trace_id)

        st.markdown(reply)
        if isinstance(result_dict, dict) and result_dict:
            _render_trace(result_dict, response_data)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "trace": result_dict if isinstance(result_dict, dict) else {},
        "result_data": response_data,
    })
