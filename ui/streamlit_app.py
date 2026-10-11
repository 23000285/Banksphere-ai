"""BankSphere AI — Streamlit banking chat interface.

Run: streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
    sources = resp.get("sources", [])
    if not answer:
        return "*No policy information available for this query.*"
    parts = [answer]
    if sources:
        parts.append("\n---\n**Sources:**")
        for s in sources:
            source_name = s.get("source", "unknown")
            category = s.get("category", "")
            heading = s.get("heading", "")
            label = f"{source_name}"
            if category:
                label += f" ({category})"
            if heading:
                label += f" — {heading}"
            parts.append(f"- 📄 {label}")
    return "\n".join(parts)


def _format_card_block(resp: dict) -> str:
    status = resp.get("status", "")
    message = resp.get("message", "")
    acct = resp.get("account_number", "")
    approval_id = resp.get("approval_request_id", "")

    if status == "pending_approval":
        lines = [
            "**Card Block Request** ⏳\n",
            f"- **Account:** `{acct}`",
            f"- **Status:** Awaiting human approval",
            f"- **Approval ID:** `{approval_id[:8]}...`" if approval_id else "",
            f"\n{message}",
            "\n> Check the **Risk & Approvals** tab or the sidebar to approve or reject this request.",
        ]
        return "\n".join(line for line in lines if line)
    elif status == "already_blocked":
        return f"**Card Block** — {message}"
    else:
        return f"**Card Block** — {message}"


def _format_response(intent: str, response: dict, trace_id: str) -> str:
    if intent == "banking":
        return _format_banking(response)
    elif intent == "fraud":
        return _format_fraud(response)
    elif intent == "credit":
        return _format_credit(response)
    elif intent == "compliance":
        return _format_compliance(response)
    elif intent == "card_block":
        return _format_card_block(response)
    return f"*Response received (trace: {trace_id[:8]})*"


def _render_trace(result_dict: dict, result_data: dict | None = None):
    agent_name = result_dict.get("agent", "unknown")
    trace_id = result_dict.get("trace_id", "")
    status = result_dict.get("status", "")
    steps = result_dict.get("steps", [])

    with st.expander(f"🔍 Agent trace — {agent_name} ({status}) [{trace_id[:8]}]"):
        if steps:
            for s in steps:
                tool = s.get("tool_name", "?")
                outcome = s.get("outcome", "?")
                dur = s.get("duration_ms", 0)
                icon = "✅" if outcome == "success" else ("🚫" if outcome == "denied" else "❌")
                st.markdown(f"{icon} **{tool}** — {outcome} ({dur:.1f}ms)")
                summary = s.get("result_summary", "")
                if summary:
                    st.caption(summary)
        else:
            st.caption("No tool steps recorded.")


def _render_risk_chart(result_data: dict):
    import pandas as pd

    scored = result_data.get("scored_transactions", [])
    if not scored:
        return

    rows = []
    for txn in scored:
        rows.append({
            "Transaction": txn.get("transaction_uid", "")[:8] + "...",
            "Amount (₹)": float(txn.get("amount", 0)),
            "Fraud Score": float(txn.get("fraud_score", 0)),
            "Risk": txn.get("risk_category", "unknown"),
            "Recommendation": txn.get("recommendation", ""),
        })

    df = pd.DataFrame(rows)
    st.subheader("Transaction Risk Scores")
    st.bar_chart(df.set_index("Transaction")["Fraud Score"], color="#ff4b4b")
    st.dataframe(df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Initialise backend
# ---------------------------------------------------------------------------

engine, fraud_artifacts, credit_artifacts, customer_map = _init_backend()


# ---------------------------------------------------------------------------
# Sidebar — customer selector
# ---------------------------------------------------------------------------

st.sidebar.title("🏦 BankSphere AI")
st.sidebar.caption("Multi-agent banking assistant")

selected_label = st.sidebar.selectbox("Select Customer", list(customer_map.keys()))
customer_uid = customer_map[selected_label]
st.sidebar.caption(f"UID: `{customer_uid}`")

st.sidebar.divider()

# Sidebar sample queries
st.sidebar.subheader("Sample Queries")
_SAMPLE_QUERIES = [
    "Show my balance",
    "What are my recent transactions?",
    "Are any of my transactions suspicious?",
    "Am I eligible for a loan?",
    "What is the KYC policy?",
    "Block my card",
]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

for q in _SAMPLE_QUERIES:
    if st.sidebar.button(q, key=f"sample_{q}", width="stretch"):
        st.session_state.pending_query = q

st.sidebar.divider()
st.sidebar.info(
    "⚠️ **Demo Mode** — This is a simulated banking environment "
    "using synthetic data. No real financial operations are performed."
)


# ---------------------------------------------------------------------------
# Approval helpers
# ---------------------------------------------------------------------------

def _render_approval_panel(customer_uid: str):
    """Render the pending approvals panel with approve/reject controls."""
    from gateway.approval import get_pending_approvals, resolve_approval

    pending = get_pending_approvals(customer_uid=customer_uid)
    if not pending:
        st.info("No pending approval requests for this customer.")
        return

    for req in pending:
        with st.container(border=True):
            col_info, col_actions = st.columns([3, 1])
            with col_info:
                st.markdown(f"**{req.action.replace('_', ' ').title()}** — `{req.request_id}`")
                st.caption(
                    f"Customer: `{req.customer_uid[:8]}...` | "
                    f"Agent: {req.agent} | "
                    f"Created: {req.created_at[:19]}"
                )
                if req.reason:
                    st.caption(f"Reason: {req.reason}")
                details = req.details
                if details.get("account_number"):
                    st.caption(f"Account: `{details['account_number']}`")
            with col_actions:
                if st.button("✅ Approve", key=f"approve_{req.request_id}", width="stretch"):
                    resolve_approval(req.request_id, approved=True, resolved_by="streamlit_user")
                    _execute_approved_action(req.request_id)
                    st.rerun()
                if st.button("❌ Reject", key=f"reject_{req.request_id}", width="stretch"):
                    resolve_approval(req.request_id, approved=False, resolved_by="streamlit_user")
                    st.rerun()


def _execute_approved_action(approval_request_id: str):
    """After approval, execute the protected action."""
    from sqlalchemy.orm import sessionmaker
    from tools.banking_tools import execute_card_block

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        result = execute_card_block(session, approval_request_id)
        if result.get("status") == "blocked":
            st.toast(f"Card blocked for account {result.get('account_number', '')}", icon="🔒")
        elif result.get("status") == "error":
            st.toast(f"Action failed: {result.get('message', '')}", icon="⚠️")
    finally:
        session.close()


def _render_approval_history(customer_uid: str):
    """Show resolved approval requests."""
    from gateway.approval import get_approval_log

    all_requests = [
        r for r in get_approval_log()
        if r.customer_uid == customer_uid and r.status.value != "pending"
    ]
    if not all_requests:
        st.caption("No resolved approvals yet.")
        return

    for req in all_requests:
        status_icon = {
            "approved": "✅",
            "rejected": "❌",
            "expired": "⏰",
        }.get(req.status.value, "❓")
        with st.container(border=True):
            st.markdown(
                f"{status_icon} **{req.action.replace('_', ' ').title()}** — "
                f"**{req.status.value.upper()}**"
            )
            st.caption(
                f"ID: `{req.request_id}` | "
                f"Resolved by: {req.resolved_by or 'N/A'} | "
                f"Resolved at: {(req.resolved_at or '')[:19]}"
            )


# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='background: #1a1a2e; padding: 0.5rem 1rem; border-radius: 8px; "
    "margin-bottom: 1rem; border-left: 4px solid #e94560;'>"
    "<span style='color: #e94560; font-weight: 600;'>⚠️ SIMULATED ENVIRONMENT</span>"
    "<span style='color: #ccc; margin-left: 0.5rem;'>— All data is synthetic. "
    "No real banking operations are performed.</span></div>",
    unsafe_allow_html=True,
)

tab_dashboard, tab_assistant, tab_risk = st.tabs([
    "📊 Banking Dashboard",
    "💬 AI Assistant",
    "🛡️ Risk & Approvals",
])


# ---------------------------------------------------------------------------
# Tab 1: Banking Dashboard
# ---------------------------------------------------------------------------

with tab_dashboard:
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select

    Session = sessionmaker(bind=engine)
    dash_session = Session()

    try:
        from ml.db_models import Customer, Account, Transaction

        customer = dash_session.execute(
            select(Customer).where(Customer.customer_uid == customer_uid)
        ).scalar_one_or_none()

        if customer is None:
            st.error("Customer not found.")
        else:
            st.subheader(f"{customer.first_name} {customer.last_name}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("KYC Status", customer.kyc_status.upper())
            col2.metric("Risk Category", customer.risk_category.upper())
            col3.metric("Accounts", len(customer.accounts))
            col4.metric("City", customer.city or "N/A")

            st.divider()

            if not customer.accounts:
                st.info("No accounts found for this customer.")
            else:
                st.subheader("Accounts")
                for acct in customer.accounts:
                    status_color = "green" if acct.status == "active" else "red"
                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                        c1.markdown(f"**{acct.account_type.title()}**")
                        c1.caption(f"`{acct.account_number}`")
                        c2.metric("Balance", f"₹{float(acct.balance):,.2f}")
                        c3.caption(f"Currency: {acct.currency}")
                        c3.caption(f"Opened: {str(acct.opened_at)[:10]}")
                        c4.markdown(
                            f"<span style='color:{status_color}; font-weight:bold'>"
                            f"● {acct.status.upper()}</span>",
                            unsafe_allow_html=True,
                        )

                st.divider()
                st.subheader("Recent Transactions")

                account_ids = [a.id for a in customer.accounts]
                acct_map = {a.id: a.account_number for a in customer.accounts}

                txns = dash_session.execute(
                    select(Transaction)
                    .where(Transaction.account_id.in_(account_ids))
                    .order_by(Transaction.timestamp.desc())
                    .limit(15)
                ).scalars().all()

                if not txns:
                    st.info("No transactions found.")
                else:
                    import pandas as pd

                    rows = []
                    for t in txns:
                        rows.append({
                            "Date": str(t.timestamp)[:10],
                            "Type": t.txn_type,
                            "Category": t.category,
                            "Amount (₹)": float(t.amount),
                            "Merchant": t.merchant or "",
                            "Channel": t.channel,
                            "Account": acct_map.get(t.account_id, ""),
                            "Flagged": "🚩" if t.is_flagged else "",
                        })

                    df = pd.DataFrame(rows)
                    st.dataframe(
                        df,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Amount (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                            "Flagged": st.column_config.TextColumn(width="small"),
                        },
                    )

                    flagged_count = sum(1 for t in txns if t.is_flagged)
                    total_debit = sum(float(t.amount) for t in txns if t.txn_type == "debit")
                    total_credit = sum(float(t.amount) for t in txns if t.txn_type == "credit")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Flagged Transactions", flagged_count)
                    m2.metric("Total Debits (shown)", f"₹{total_debit:,.2f}")
                    m3.metric("Total Credits (shown)", f"₹{total_credit:,.2f}")
    finally:
        dash_session.close()


# ---------------------------------------------------------------------------
# Tab 2: AI Assistant
# ---------------------------------------------------------------------------

with tab_assistant:
    st.subheader("Banking Agent Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            "Welcome! Ask me about your accounts, transactions, loans, "
            "fraud checks, or banking policies. Try one of the sample queries "
            "in the sidebar to get started."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                _render_trace(msg["trace"], msg.get("result_data"))
                if msg.get("result_data", {}).get("scored_transactions"):
                    _render_risk_chart(msg["result_data"])

    # Handle sample query from sidebar
    active_query = None
    if st.session_state.pending_query:
        active_query = st.session_state.pending_query
        st.session_state.pending_query = None

    if prompt := (active_query or st.chat_input("Ask about your account, transactions, loans, or policies...")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from sqlalchemy.orm import sessionmaker as sm
                from agents.supervisor import RoutingRequest, run_supervisor
                from gateway.pii import mask_pii

                ChatSession = sm(bind=engine)
                session = ChatSession()
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
                reply = f"**Request rejected:** {mask_pii(result.error or '')}"
            elif result.error:
                reply = f"**Error:** {mask_pii(result.error)}"
            elif result.intent == "unsupported":
                reply = (
                    "I'm not sure how to help with that. I can assist with "
                    "banking queries, fraud checks, loan eligibility, policy questions, "
                    "and card management."
                )
            else:
                reply = _format_response(result.intent, response_data, result.trace_id)

            st.markdown(reply)
            if isinstance(result_dict, dict) and result_dict:
                _render_trace(result_dict, response_data)
                if response_data.get("scored_transactions"):
                    _render_risk_chart(response_data)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "trace": result_dict if isinstance(result_dict, dict) else {},
            "result_data": response_data,
        })

        if active_query:
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 3: Risk & Approvals
# ---------------------------------------------------------------------------

with tab_risk:
    risk_col, approval_col = st.columns(2)

    with risk_col:
        st.subheader("🔍 Risk Analytics")
        st.caption(
            "Run a fraud check from the AI Assistant tab to see risk scores here. "
            "Ask: \"Are any of my transactions suspicious?\""
        )

        last_fraud = None
        for msg in reversed(st.session_state.get("messages", [])):
            rd = msg.get("result_data", {})
            if rd.get("scored_transactions"):
                last_fraud = rd
                break

        if last_fraud:
            _render_risk_chart(last_fraud)
        else:
            st.info("No fraud analysis results yet. Use the AI Assistant to check for suspicious transactions.")

    with approval_col:
        st.subheader("🔐 Approval Queue")
        _render_approval_panel(customer_uid)

        st.divider()
        st.subheader("📋 Approval History")
        _render_approval_history(customer_uid)
