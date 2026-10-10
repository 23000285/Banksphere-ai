# BankSphere AI — Architecture

## System Overview

BankSphere AI uses a supervisor-agent pattern where a central supervisor classifies user intent and routes requests to specialist agents, each with restricted tool access and validated inputs/outputs.

The reference architecture (see `Reference Image - AI Task.jpeg`) defines a nine-layer stack for an enterprise banking AI platform. This MVP implements a vertical slice through those layers, focusing on the multi-agent orchestrator, tool layer, and data layer with synthetic data. The sections below describe each reference layer and note the current implementation status after Day 1 (Phases 0–7).

## Reference Architecture Layers

### Layer 1 — Banking Channels / Users

> **Status:** Not implemented (Day 2 — Phases 11–12 will add API and Streamlit UI)

The reference architecture shows six user-facing channels:

| Channel | Description |
|---|---|
| Mobile App | Native mobile banking application |
| Web Banking | Browser-based internet banking portal |
| Branch | In-branch banker-assisted interface |
| Call Center | Telephony / IVR integration |
| RM Portal | Relationship manager dashboard |
| API Partners | Third-party integrations via open APIs |

**MVP scope:** Phase 12 will add a single Streamlit web interface as the user channel. Phase 11 will add a FastAPI endpoint that could serve as the API Partners channel. The remaining channels (Mobile, Branch, Call Center, RM Portal) are outside MVP scope.

### Layer 2 — API & Experience Layer

> **Status:** Not implemented (Day 2 — Phase 11)

The reference architecture defines an API gateway with security and observability:

| Component | Purpose |
|---|---|
| API Gateway | Central entry point for all client requests |
| WAF | Web Application Firewall for request filtering |
| OAuth2 / OIDC | Authentication and identity federation |
| Rate Limiting | Throttle requests per client/endpoint |
| Session Management | Maintain user session state |
| Request Validation | Schema and payload validation at the edge |
| Threat Protection | Block known attack patterns |
| Correlation / Trace ID | Assign unique IDs for end-to-end tracing |

**MVP scope:** Phase 11 will add FastAPI with Pydantic request validation, basic API-key authentication, and trace IDs. WAF, OAuth2/OIDC, rate limiting, session management, and threat protection are outside MVP scope.

### Layer 3 — AI Gateway / AI Control Plane

> **Status:** Partially implemented

The reference architecture defines an AI-specific control plane with eleven subcomponents:

| Component | Reference Purpose | MVP Status |
|---|---|---|
| Prompt Management | Template management, versioning, injection | **Not implemented** — no prompt templating system |
| Model Routing | Select model per request type | **Implemented** — `gateway/llm_config.py` routes to mock/Gemini/Bedrock via LiteLLM |
| Token Budget | Per-request and per-agent token limits | **Not implemented** |
| Guardrails | Input/output safety checks | **Not implemented** — planned for Phase 10 |
| PII Detection | Detect and mask personal data in prompts/responses | **Not implemented** — planned for Phase 10 |
| Policy Enforcement | Apply organizational policies to LLM interactions | **Not implemented** — planned for Phase 10 |
| Content Safety | Filter harmful or inappropriate content | **Not implemented** — planned for Phase 10 |
| Model Selection | Choose model based on capability/cost | **Implemented** — `gateway/llm_config.py` supports provider switching (mock/gemini/bedrock) |
| Cost Controls | Budget limits and cost tracking per request | **Not implemented** |
| Audit Policy | Log all LLM interactions for compliance | **Partially implemented** — `gateway/audit.py` logs tool invocations with trace IDs; LLM call-level audit not yet added |
| Human-in-the-Loop | Pause execution for human review/approval | **Not implemented** — fraud agent returns `requires_approval: true` flag but execution does not pause; blocking approval workflow planned for Phase 10 |

**Currently implemented in `gateway/`:**
- `llm_config.py` — `LLMConfig` dataclass with provider validation (mock/gemini/bedrock)
- `llm_provider.py` — Factory returning `MockLLM` or `LiteLLMWrapper` via LiteLLM
- `mock_llm.py` — Deterministic mock with `[MOCK]` prefix and keyword-based intent matching
- `audit.py` — `AuditEvent` dataclass with structured logging (trace_id, tool, agent, requester, target, outcome, duration_ms); `AuditTimer` context manager

### Layer 4 — Multi-Agent Orchestrator

> **Status:** Implemented (Phases 5–7)

The reference architecture defines a supervisor/agent orchestrator with eight responsibilities and nine specialist agents.

#### Supervisor Responsibilities

| Responsibility | Reference Description | MVP Status |
|---|---|---|
| Intent Classification | Determine what the user is asking | **Implemented** — keyword-based regex matching in `agents/supervisor.py:classify_intent()` |
| Task Decomposition | Break complex requests into sub-tasks | **Not implemented** — single-intent routing only |
| Agent Selection | Choose which specialist to invoke | **Implemented** — conditional edges in LangGraph StateGraph |
| Workflow Management | Orchestrate multi-step agent workflows | **Partially implemented** — single classify → specialist → END flow via LangGraph |
| State Management | Track state across the workflow | **Implemented** — `_GraphState` TypedDict passed through LangGraph nodes |
| Error Recovery | Handle agent failures gracefully | **Implemented** — agent errors propagate to `RoutingResult.error`; agents catch tool exceptions and record them as steps |
| Result Aggregation | Combine specialist outputs for the user | **Implemented** — specialist result embedded in `RoutingResult.result` |
| Human Escalation | Escalate to a human when confidence is low | **Not implemented** — unsupported queries return an error message; no escalation path |

**Implementation:** `agents/supervisor.py` uses a LangGraph `StateGraph` with five nodes (classify, banking, fraud, credit, unsupported) and conditional routing edges. Input validated via `RoutingRequest` (Pydantic), output via `RoutingResult` (Pydantic). Execution is sequential — one intent per request.

#### Specialist Agents

| Agent | Reference Capabilities | MVP Status |
|---|---|---|
| **Customer Agent** | Customer Profile, Account Info, Transaction History, Customer 360 | **Implemented as Banking Agent** — `agents/banking_agent.py` provides profile, accounts, transactions |
| **Banking Product Agent** | Savings & Accounts, Cards, Loans, Investments, Insurance | **Partially implemented** — banking agent covers accounts and card status; loans handled by credit agent; investments and insurance are outside MVP scope |
| **Risk / Fraud Agent** | Transaction Monitoring, Anomaly Detection, Device/Geo Analysis, Fraud Risk Scoring | **Implemented** — `agents/fraud_agent.py` with LightGBM classifier, Isolation Forest anomaly detection, SHAP explanations; returns allow/verify/block recommendation |
| **Credit Agent** | Credit Bureau Data, Eligibility Check, Risk Features, Loan Policy Retrieval | **Implemented** — `agents/credit_agent.py` with GradientBoosting model, deterministic eligibility rules, EMI calculator; no external credit bureau or loan policy retrieval |
| **Compliance Agent** | KYC/AML, Regulatory Policies, Internal Policies, Compliance Checks | **Not implemented** — planned for Phase 9; permission entry exists in `AGENT_PERMISSIONS` (`retrieve_policy`) |
| **Financial Advisory Agent** | Financial Planning, Investment Guidance, Personalized Insights, Wealth Management | **Not implemented** — outside MVP scope |
| **Research Agent** | Market Research, Economic Analysis, Competitor Insights, Product Research | **Not implemented** — outside MVP scope |
| **Document Agent** | Document Processing, Data Extraction, Verification, Summarization | **Not implemented** — outside MVP scope |
| **Notification Agent** | Customer Alerts, Communication, Multi-channel, Event Triggers | **Not implemented** — notification stub planned for Phase 9 |

**Base agent framework:** `agents/base.py` defines `BaseAgent` (dataclass with max_steps, trace_id, step recording), `AgentStep` (Pydantic: step number, tool_name, outcome, result_summary, duration_ms), and `AgentResult` (Pydantic: agent name, trace_id, status, steps list, response, error). All three implemented agents (banking, fraud, credit) extend `BaseAgent`.

### Layer 5 — Agent Tool / MCP Layer

> **Status:** Partially implemented (Phases 3–6)

The reference architecture defines a tool layer with eight infrastructure components and twelve domain tool APIs.

#### Tool Infrastructure

| Component | Reference Purpose | MVP Status |
|---|---|---|
| Tool Registry | Central catalog of available tools | **Implemented** — `tools/registry.py` with `_TOOL_REGISTRY` dict, `register_tool()`, `invoke_tool()` |
| MCP Gateway | Model Context Protocol gateway for tool access | **Not implemented** — outside MVP scope |
| Authorization | Verify agent and user permissions before tool execution | **Implemented** — `invoke_tool()` checks `AGENT_PERMISSIONS` map; banking tools enforce customer-level auth via `_authorize()` |
| Schema Validation | Validate tool inputs and outputs | **Implemented** — `tools/schemas.py` with 9 Pydantic v2 models for all tool inputs/outputs |
| Tool Sandbox | Isolate tool execution environments | **Not implemented** — outside MVP scope |
| Permission Boundary | Enforce least-privilege tool access per agent | **Implemented** — `AGENT_PERMISSIONS` map restricts each agent to its allowed tool set |
| Transaction Approval | Require approval for sensitive operations | **Not implemented** — `requires_approval` flag exists in fraud responses but no blocking gate; planned for Phase 10 |
| Audit Logging | Record all tool invocations | **Implemented** — `gateway/audit.py` `AuditEvent` logged for every banking tool call with trace_id, timing, and outcome |

#### Domain Tool APIs

| Tool API | Reference Purpose | MVP Status |
|---|---|---|
| Account API | Account queries and operations | **Implemented** — `get_accounts()` in `tools/banking_tools.py` |
| Loan API | Loan operations | **Implemented** — `calculate_emi()` and `assess_credit_risk()` in `ml/credit_model.py` |
| Credit API | Credit scoring and assessment | **Implemented** — `assess_credit_risk()` in `ml/credit_model.py` |
| Payment API | Payment processing | **Not implemented** — outside MVP scope |
| CRM | Customer relationship management | **Not implemented** — outside MVP scope |
| KYC | Know Your Customer verification | **Not implemented** — planned for Phase 9 (compliance agent) |
| AML | Anti-money laundering checks | **Not implemented** — outside MVP scope |
| Card API | Card status and operations | **Implemented** — `get_card_status()` in `tools/banking_tools.py` (read-only) |
| Transaction API | Transaction queries | **Implemented** — `get_transactions()` in `tools/banking_tools.py` |
| Notification | Customer notification dispatch | **Not implemented** — stub planned for Phase 9 |
| Document | Document processing | **Not implemented** — outside MVP scope |
| Core Banking | Core banking system integration | **Not implemented** — synthetic SQLite database used instead |

**Registered tools (7 total):**
- Banking: `get_customer_profile`, `get_accounts`, `get_transactions`, `get_card_status`
- Fraud: `score_transaction`
- Credit: `assess_credit_risk`, `calculate_emi`
- Compliance: `retrieve_policy` (permission defined, tool not yet registered — Phase 9)

### Layer 6 — Enterprise Data

> **Status:** Implemented with synthetic data (Phase 2)

The reference architecture shows six enterprise data sources:

| Data Source | Reference Purpose | MVP Status |
|---|---|---|
| Core Banking | Primary banking system of record | **Simulated** — SQLite database with synthetic data via `ml/seed_data.py` |
| CRM | Customer relationship data | **Simulated** — Customer model in `ml/db_models.py` |
| Transactions | Transaction ledger | **Simulated** — Transaction model with ~50K synthetic records |
| Customer 360 | Unified customer view | **Partially simulated** — Customer profile tool aggregates from Customer and Account models |
| Loan Systems | Loan origination and servicing | **Simulated** — Loan model in `ml/db_models.py` |
| Other Internal Systems | Additional enterprise data | **Not applicable** |

**Implementation:** SQLAlchemy 2.x ORM with four models in `ml/db_models.py`:
- `Customer` — customer_uid, name, email, phone, dob, address, city, state, kyc_status, risk_category
- `Account` — account_number, customer_id (FK), account_type (savings/current/salary), balance (INR), status
- `Transaction` — transaction_uid, account_id (FK), txn_type, amount, channel, merchant, is_flagged, timestamp
- `Loan` — loan_uid, customer_id (FK), loan_type, principal, interest_rate, emi, credit_score, status

Deterministic generation via `ml/seed_data.py` with Faker + NumPy (seed=42): ~200 customers, ~50K transactions, Indian cities/states, `@banksphere.test` emails, INR currency.

### Layer 7 — Knowledge / RAG

> **Status:** Not implemented (Day 2 — Phases 8–9)

The reference architecture defines a knowledge layer with six document categories and a retrieval pipeline:

| Component | Reference Purpose | MVP Status |
|---|---|---|
| Policy Docs | Banking policy documents | **Not implemented** — Phase 8 will create 5–6 synthetic policy files |
| SOPs | Standard operating procedures | **Not implemented** — Phase 8 |
| Regulations | Regulatory requirements | **Not implemented** — Phase 8 |
| Product Docs | Product information and guides | **Not implemented** — Phase 8 |
| Compliance Docs | Compliance documentation | **Not implemented** — Phase 8 |
| Internal Knowledge | Organizational knowledge base | **Not implemented** — outside MVP scope |
| Vector DB + BM25 | Hybrid retrieval (dense + sparse) | **Not implemented** — Phase 8 will add ChromaDB + BM25 |
| Embeddings + Reranker | Embedding model and result reranking | **Not implemented** — Phase 8; reranker is optional per CLAUDE.md |

**MVP plan:** Phase 8 will add a small synthetic policy corpus with ChromaDB vector storage and BM25 retrieval. Phase 9 will add the compliance agent that queries this layer. The `rag/` directory exists with an empty `__init__.py`.

### Layer 8 — External Services

> **Status:** Not implemented — outside MVP scope

The reference architecture shows six categories of external service integrations:

| Service | Reference Purpose | MVP Status |
|---|---|---|
| Credit Bureau | External credit scoring and reports | **Not implemented** — credit scoring uses internal synthetic model |
| Payment Networks | Card networks, payment processors | **Not implemented** |
| Market Data | Market prices, economic indicators | **Not implemented** |
| KYC / AML Services | Identity verification, sanctions screening | **Not implemented** |
| Government APIs | Government data services | **Not implemented** |
| Other Third-party Services | Additional external integrations | **Not implemented** |

All external service interactions are replaced by synthetic data and internal models in this MVP.

### Layer 9 — Cross-Cutting Production Platform

> **Status:** Partially implemented

The reference architecture defines seven cross-cutting concerns:

| Concern | Reference Subcomponents | MVP Status |
|---|---|---|
| **Observability** | Traces, Logs, Metrics, Agent trajectories | **Partially implemented** — Python `logging` module used for audit events; `trace_id` propagated through agents and supervisor; `AgentStep` records tool timing; no metrics collection or distributed tracing |
| **Evaluation** | Accuracy, Groundedness, Hallucination, Retrieval Quality, Model Drift | **Not implemented** — planned for Phase 13; fraud holdout metrics (PR-AUC, precision, recall) computed at training time |
| **Security** | Identity & Access, Data Protection, PII Masking, Threat Detection, Zero Trust | **Partially implemented** — customer-level authorization in tools, per-agent permission enforcement; PII masking and threat detection planned for Phase 10 |
| **Governance** | Policy Management, Compliance, Model Governance, Risk Management | **Not implemented** — outside MVP scope |
| **Audit** | Audit Logs, Traceability, Regulatory Reporting, Incident Response | **Partially implemented** — `AuditEvent` logs tool, agent, requester, target, outcome, trace_id, and duration_ms; no regulatory reporting or incident response |
| **FinOps** | Token Usage, Cost Monitoring, Budget Controls, Resource Optimization | **Not implemented** — outside MVP scope |
| **Performance & Reliability** | P50/P95/P99, Error Rate, Tool Failures, SLA Monitoring, Incident Management | **Not implemented** — outside MVP scope |

## Directory Structure

```
Banksphere-ai/
├── app/                 # FastAPI entry point and API schemas
│   └── config.py        # Pydantic BaseSettings with .env loading
├── ui/                  # Streamlit application (stub — Phase 12)
├── agents/              # Supervisor and specialist agents
│   ├── base.py          # BaseAgent, AgentStep, AgentResult
│   ├── banking_agent.py # Balance, transactions, profile
│   ├── fraud_agent.py   # Transaction scoring, recommendations
│   ├── credit_agent.py  # Credit risk, eligibility, EMI
│   └── supervisor.py    # LangGraph StateGraph routing
├── tools/               # Tool registry and domain-specific tools
│   ├── registry.py      # Central registry, permissions, invoke_tool
│   ├── banking_tools.py # 4 banking tools with auth and audit
│   └── schemas.py       # Pydantic input/output schemas
├── ml/                  # Data generation, training, model artifacts
│   ├── db_models.py     # SQLAlchemy ORM: Customer, Account, Transaction, Loan
│   ├── db_engine.py     # Engine creation, table init, session factory
│   ├── seed_data.py     # Faker/NumPy synthetic data generator
│   ├── fraud_model.py   # LightGBM + Isolation Forest + SHAP
│   └── credit_model.py  # GradientBoosting + eligibility rules + EMI
├── gateway/             # LLM config, mock mode, audit
│   ├── llm_config.py    # LLMConfig with provider validation
│   ├── llm_provider.py  # Factory: MockLLM or LiteLLMWrapper
│   ├── mock_llm.py      # Deterministic mock with intent matching
│   └── audit.py         # AuditEvent and AuditTimer
├── rag/                 # Synthetic policy documents, ingestion, retrieval (stub — Phase 8)
├── data/                # Generated local database (not committed)
├── eval/                # Evaluation queries and runner (stub — Phase 13)
├── tests/               # Unit and integration tests (218 tests)
├── CLAUDE.md            # Project guide for Claude Code
├── README.md            # Project documentation
├── ARCHITECTURE.md      # This file
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template (no real secrets)
└── .gitignore           # Excludes secrets, databases, model artifacts
```

## Agent Responsibilities

| Agent | Role | Allowed Tools | Implementation |
|---|---|---|---|
| **Supervisor** | Intent classification, routing, result aggregation | Routing only — no direct DB or tool access | `agents/supervisor.py` — LangGraph StateGraph |
| **Banking** | Customer profiles, accounts, transactions, card status | `get_customer_profile`, `get_accounts`, `get_transactions`, `get_card_status` | `agents/banking_agent.py` |
| **Fraud** | Transaction scoring, anomaly detection, risk recommendation | `get_transactions`, `score_transaction` | `agents/fraud_agent.py` |
| **Credit** | Credit risk, eligibility rules, EMI calculation | `get_customer_profile`, `assess_credit_risk`, `calculate_emi` | `agents/credit_agent.py` |
| **Compliance** | Policy Q&A from retrieved evidence with citations | `retrieve_policy` | Not implemented — Phase 9 |

## Security Boundaries

### Implemented (Phases 0–7)

- **Customer-level authorization** enforced in tools (`_authorize()` in `banking_tools.py`), not just prompts — cross-customer access raises `AuthorizationDenied` with audit event
- **Per-agent tool permission enforcement** via `AGENT_PERMISSIONS` map in `tools/registry.py` — `invoke_tool()` denies unauthorized calls before execution
- **Pydantic input/output validation** on all tool schemas, routing models, and ML request/response models
- **Structured audit logging** via `gateway/audit.py` — records tool, agent, requester, target, outcome, trace_id, and duration_ms without secrets or PII
- **Agent step limits** — `BaseAgent._check_step_limit()` prevents unbounded agent loops
- **Trace ID propagation** — generated at agent and supervisor level, carried through all results
- **Secrets excluded from Git** — `.env` gitignored; `.env.example` has placeholders only; `api_secret_key` has `repr=False`
- **All data is synthetic** — clearly disclosed in code and documentation; `is_synthetic: true` on ML responses

### Not Yet Implemented (Planned for Day 2)

- **PII masking** — no PII detection or masking in gateway or tools (Phase 10)
- **Prompt-injection detection** — no input sanitization beyond Pydantic validation (Phase 10)
- **Human approval gate** — fraud agent returns `requires_approval: true` flag but execution does not pause; blocking workflow planned for Phase 10
- **Content safety checks** — no output filtering (Phase 10)
- **API-level authentication** — no FastAPI endpoints yet (Phase 11)
- **Rate limiting** — no request throttling (Phase 11)

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Python version** | 3.12.3 | CLAUDE.md targets 3.11; 3.11 unavailable in environment; all deps compatible |
| **LLM mode** | Mock by default | Zero-cost-first; `MockLLM` with `[MOCK]` prefix ensures offline development; real LLM optional via config |
| **Intent classification** | Keyword regex, not LLM | Deterministic, fast, no API cost; fraud keywords checked before banking to avoid ambiguity |
| **Agent orchestration** | LangGraph StateGraph | Single classify → specialist → END flow; TypedDict internal state with Pydantic at entry/exit |
| **Execution model** | Sequential, not parallel | Simpler debugging; avoids rate-limit issues with free-tier LLM providers |
| **ML training** | In-memory, no persistence | Models trained from synthetic datasets each run; no `.pkl`/`.joblib` files to manage |
| **Fraud recommendation** | Flag-only, not blocking | `requires_approval` is a response field; no execution pause; safer for MVP without approval workflow |
| **Database** | SQLite in-memory for tests, file for dev | Fast test execution; deterministic seed (42) for reproducibility |
| **Synthetic data locale** | India (cities, INR, pin codes) | Realistic but clearly artificial; `@banksphere.test` email domain |
