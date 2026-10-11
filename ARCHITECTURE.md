# BankSphere AI — Architecture

## System Overview

BankSphere AI uses a supervisor-agent pattern where a central supervisor classifies user intent and routes requests to specialist agents, each with restricted tool access and validated inputs/outputs.

The reference architecture (see `Reference Image - AI Task.jpeg`) defines a nine-layer stack for an enterprise banking AI platform. This MVP implements a vertical slice through those layers, focusing on the multi-agent orchestrator, tool layer, and data layer with synthetic data. The sections below describe each reference layer and note the current implementation status after all phases (0–14).

## Reference Architecture Layers

### Layer 1 — Banking Channels / Users

> **Status:** Implemented (Phases 11–12)

The reference architecture shows six user-facing channels:

| Channel | Description |
|---|---|
| Mobile App | Native mobile banking application |
| Web Banking | Browser-based internet banking portal |
| Branch | In-branch banker-assisted interface |
| Call Center | Telephony / IVR integration |
| RM Portal | Relationship manager dashboard |
| API Partners | Third-party integrations via open APIs |

**MVP scope:** Streamlit web interface (`ui/streamlit_app.py`) serves as the user channel with Banking Dashboard, AI Assistant, and Risk & Approvals tabs. FastAPI (`app/main.py`) serves as the API Partners channel. The remaining channels (Mobile, Branch, Call Center, RM Portal) are outside MVP scope.

### Layer 2 — API & Experience Layer

> **Status:** Implemented (Phase 11)

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

**MVP scope:** FastAPI with Pydantic request validation, API-key authentication (`X-API-Key`), and trace IDs (`X-Trace-ID`). WAF, OAuth2/OIDC, rate limiting, session management, and threat protection are outside MVP scope.

### Layer 3 — AI Gateway / AI Control Plane

> **Status:** Partially implemented

The reference architecture defines an AI-specific control plane with eleven subcomponents:

| Component | Reference Purpose | MVP Status |
|---|---|---|
| Prompt Management | Template management, versioning, injection | **Not implemented** — no prompt templating system |
| Model Routing | Select model per request type | **Implemented** — `gateway/llm_config.py` routes to mock/Gemini/Bedrock via LiteLLM |
| Token Budget | Per-request and per-agent token limits | **Not implemented** |
| Guardrails | Input/output safety checks | **Implemented** — `gateway/input_guard.py` detects prompt injection, role assumption, data exfiltration, SQL injection |
| PII Detection | Detect and mask personal data in prompts/responses | **Implemented** — `gateway/pii.py` masks Aadhaar, PAN, email, phone, account numbers in responses |
| Policy Enforcement | Apply organizational policies to LLM interactions | **Implemented** — per-agent tool permissions, customer-level auth, sensitive action approval gate |
| Content Safety | Filter harmful or inappropriate content | **Implemented** — input guard rejects harmful patterns; PII masked on output |
| Model Selection | Choose model based on capability/cost | **Implemented** — `gateway/llm_config.py` supports provider switching (mock/gemini/bedrock) |
| Cost Controls | Budget limits and cost tracking per request | **Not implemented** |
| Audit Policy | Log all LLM interactions for compliance | **Partially implemented** — `gateway/audit.py` logs tool invocations with trace IDs; LLM call-level audit not yet added |
| Human-in-the-Loop | Pause execution for human review/approval | **Implemented** — `gateway/approval.py` blocks sensitive actions (card_block, fund_transfer, etc.) until explicit human approval; time-based expiration enforced |

**Currently implemented in `gateway/`:**
- `llm_config.py` — `LLMConfig` dataclass with provider validation (mock/gemini/bedrock)
- `llm_provider.py` — Factory returning `MockLLM` or `LiteLLMWrapper` via LiteLLM
- `mock_llm.py` — Deterministic mock with `[MOCK]` prefix and keyword-based intent matching

**LLM narrative generation (`agents/response_generator.py`):**
- After each specialist agent completes, the supervisor calls `generate_narrative()` which uses the configured LLM provider to produce a natural-language explanation of the verified tool results
- In mock mode, narratives are `[MOCK]`-prefixed; in Gemini mode, they are real LLM-generated text
- The LLM receives only PII-masked, structured data — it cannot invent balances, scores, or citations
- Provider failures produce an empty narrative; the system falls back to the existing structured display
- `audit.py` — `AuditEvent` dataclass with structured logging (trace_id, tool, agent, requester, target, outcome, duration_ms); `AuditTimer` context manager
- `input_guard.py` — Prompt injection detection (role assumption, instruction override, system prompt leak, delimiter injection, data exfiltration, SQL injection)
- `pii.py` — PII masking for Aadhaar, PAN, email, phone, account numbers
- `approval.py` — Human approval gate with PENDING/APPROVED/REJECTED/EXPIRED states and time-based expiration (configurable, default 15 min)
- `notifications.py` — Notification stub for simulated alerts (no real messages sent)

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

**Implementation:** `agents/supervisor.py` uses a LangGraph `StateGraph` with seven nodes (classify, banking, fraud, credit, compliance, card_block, unsupported) and conditional routing edges. Input validated via `RoutingRequest` (Pydantic), output via `RoutingResult` (Pydantic). Input guard checks run before routing. Execution is sequential — one intent per request.

#### Specialist Agents

| Agent | Reference Capabilities | MVP Status |
|---|---|---|
| **Customer Agent** | Customer Profile, Account Info, Transaction History, Customer 360 | **Implemented as Banking Agent** — `agents/banking_agent.py` provides profile, accounts, transactions |
| **Banking Product Agent** | Savings & Accounts, Cards, Loans, Investments, Insurance | **Partially implemented** — banking agent covers accounts and card status; loans handled by credit agent; investments and insurance are outside MVP scope |
| **Risk / Fraud Agent** | Transaction Monitoring, Anomaly Detection, Device/Geo Analysis, Fraud Risk Scoring | **Implemented** — `agents/fraud_agent.py` with LightGBM classifier, Isolation Forest anomaly detection, SHAP explanations; returns allow/verify/block recommendation |
| **Credit Agent** | Credit Bureau Data, Eligibility Check, Risk Features, Loan Policy Retrieval | **Implemented** — `agents/credit_agent.py` with GradientBoosting model, deterministic eligibility rules, EMI calculator; no external credit bureau or loan policy retrieval |
| **Compliance Agent** | KYC/AML, Regulatory Policies, Internal Policies, Compliance Checks | **Implemented** — `agents/compliance_agent.py` answers from retrieved policy evidence with citations; declines when evidence is absent |
| **Financial Advisory Agent** | Financial Planning, Investment Guidance, Personalized Insights, Wealth Management | **Not implemented** — outside MVP scope |
| **Research Agent** | Market Research, Economic Analysis, Competitor Insights, Product Research | **Not implemented** — outside MVP scope |
| **Document Agent** | Document Processing, Data Extraction, Verification, Summarization | **Not implemented** — outside MVP scope |
| **Notification Agent** | Customer Alerts, Communication, Multi-channel, Event Triggers | **Implemented** — `gateway/notifications.py` provides structured simulated notifications (no real messages sent) |

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
| Transaction Approval | Require approval for sensitive operations | **Implemented** — `gateway/approval.py` enforces human approval for card_block, fund_transfer, account_closure, limit_increase; `execute_card_block()` checks approval status with time-based expiration |
| Audit Logging | Record all tool invocations | **Implemented** — `gateway/audit.py` `AuditEvent` logged for every banking tool call with trace_id, timing, and outcome |

#### Domain Tool APIs

| Tool API | Reference Purpose | MVP Status |
|---|---|---|
| Account API | Account queries and operations | **Implemented** — `get_accounts()` in `tools/banking_tools.py` |
| Loan API | Loan operations | **Implemented** — `calculate_emi()` and `assess_credit_risk()` in `ml/credit_model.py` |
| Credit API | Credit scoring and assessment | **Implemented** — `assess_credit_risk()` in `ml/credit_model.py` |
| Payment API | Payment processing | **Not implemented** — outside MVP scope |
| CRM | Customer relationship management | **Not implemented** — outside MVP scope |
| KYC | Know Your Customer verification | **Implemented** — compliance agent answers KYC policy questions from retrieved evidence |
| AML | Anti-money laundering checks | **Not implemented** — outside MVP scope |
| Card API | Card status and operations | **Implemented** — `get_card_status()` in `tools/banking_tools.py` (read-only) |
| Transaction API | Transaction queries | **Implemented** — `get_transactions()` in `tools/banking_tools.py` |
| Notification | Customer notification dispatch | **Implemented** — `gateway/notifications.py` stub logs structured events (no real messages) |
| Document | Document processing | **Not implemented** — outside MVP scope |
| Core Banking | Core banking system integration | **Not implemented** — synthetic SQLite database used instead |

**Registered tools (8 total):**
- Banking: `get_customer_profile`, `get_accounts`, `get_transactions`, `get_card_status`, `request_card_block`
- Fraud: `score_transaction`
- Credit: `assess_credit_risk`, `calculate_emi`
- Compliance: `retrieve_policy` (hybrid RAG backed by ChromaDB + BM25)

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

> **Status:** Implemented (Phases 8–9)

The reference architecture defines a knowledge layer with six document categories and a retrieval pipeline:

| Component | Reference Purpose | MVP Status |
|---|---|---|
| Policy Docs | Banking policy documents | **Implemented** — 6 synthetic policy files in `rag/policies/` (KYC, loans, fraud, cards, minor accounts, dormant accounts) |
| SOPs | Standard operating procedures | **Partially covered** — policy docs include procedural content |
| Regulations | Regulatory requirements | **Partially covered** — KYC and compliance policies reference regulatory requirements |
| Product Docs | Product information and guides | **Not implemented** — outside MVP scope |
| Compliance Docs | Compliance documentation | **Implemented** — compliance policies with AML/KYC content |
| Internal Knowledge | Organizational knowledge base | **Not implemented** — outside MVP scope |
| Vector DB + BM25 | Hybrid retrieval (dense + sparse) | **Implemented** — `rag/chroma_store.py` (ChromaDB) + `rag/bm25_retriever.py` (rank-bm25) |
| Embeddings + Reranker | Embedding model and result reranking | **Implemented** — ChromaDB default embeddings; Reciprocal Rank Fusion (RRF) in `rag/hybrid_retriever.py` for merging; no separate reranker |

**Implementation:** `rag/hybrid_retriever.py` combines ChromaDB vector search and BM25 keyword search using RRF scoring. Policy documents are loaded from markdown files, chunked by heading, and ingested into both stores. The compliance agent queries via `tools/compliance_tools.py`.

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
| **Evaluation** | Accuracy, Groundedness, Hallucination, Retrieval Quality, Model Drift | **Implemented** — `eval/` contains evaluation queries and runner; fraud holdout metrics (PR-AUC, precision, recall) computed at training time |
| **Security** | Identity & Access, Data Protection, PII Masking, Threat Detection, Zero Trust | **Implemented** — customer-level authorization in tools, per-agent permission enforcement, PII masking on responses, prompt injection detection, approval gate for sensitive actions |
| **Governance** | Policy Management, Compliance, Model Governance, Risk Management | **Not implemented** — outside MVP scope |
| **Audit** | Audit Logs, Traceability, Regulatory Reporting, Incident Response | **Partially implemented** — `AuditEvent` logs tool, agent, requester, target, outcome, trace_id, and duration_ms; no regulatory reporting or incident response |
| **FinOps** | Token Usage, Cost Monitoring, Budget Controls, Resource Optimization | **Not implemented** — outside MVP scope |
| **Performance & Reliability** | P50/P95/P99, Error Rate, Tool Failures, SLA Monitoring, Incident Management | **Not implemented** — outside MVP scope |

## Directory Structure

```
Banksphere-ai/
├── app/                 # FastAPI entry point and API schemas
│   ├── main.py          # FastAPI app with /chat, /approval, /health endpoints
│   └── config.py        # Pydantic BaseSettings with .env loading
├── ui/                  # Streamlit application
│   └── streamlit_app.py # 3-tab UI: Dashboard, AI Assistant, Risk & Approvals
├── agents/              # Supervisor and specialist agents
│   ├── base.py          # BaseAgent, AgentStep, AgentResult
│   ├── banking_agent.py # Balance, transactions, profile, card block
│   ├── fraud_agent.py   # Transaction scoring, recommendations
│   ├── credit_agent.py  # Credit risk, eligibility, EMI
│   ├── compliance_agent.py # Policy Q&A with citations and refusal
│   ├── response_generator.py # LLM narrative generation from verified results
│   └── supervisor.py    # LangGraph StateGraph routing with input guard
├── tools/               # Tool registry and domain-specific tools
│   ├── registry.py      # Central registry, permissions, invoke_tool
│   ├── banking_tools.py # 5 banking tools with auth, audit, approval gate
│   ├── compliance_tools.py # Policy retrieval backed by hybrid RAG
│   └── schemas.py       # Pydantic input/output schemas
├── ml/                  # Data generation, training, model artifacts
│   ├── db_models.py     # SQLAlchemy ORM: Customer, Account, Transaction, Loan
│   ├── db_engine.py     # Engine creation, table init, session factory
│   ├── seed_data.py     # Faker/NumPy synthetic data generator
│   ├── fraud_model.py   # LightGBM + Isolation Forest + SHAP
│   └── credit_model.py  # GradientBoosting + eligibility rules + EMI
├── gateway/             # Security, guardrails, approval, audit
│   ├── llm_config.py    # LLMConfig with provider validation
│   ├── llm_provider.py  # Factory: MockLLM or LiteLLMWrapper
│   ├── mock_llm.py      # Deterministic mock with intent matching
│   ├── audit.py         # AuditEvent and AuditTimer
│   ├── input_guard.py   # Prompt injection detection
│   ├── pii.py           # PII masking (Aadhaar, PAN, email, phone, accounts)
│   ├── approval.py      # Human approval gate with time-based expiration
│   └── notifications.py # Notification stub (no real messages)
├── rag/                 # Hybrid retrieval pipeline
│   ├── loader.py        # Policy document loader and chunker
│   ├── chroma_store.py  # ChromaDB vector store
│   ├── bm25_retriever.py # BM25 keyword retriever
│   ├── hybrid_retriever.py # RRF merging of dense + sparse results
│   └── policies/        # 6 synthetic policy markdown files
├── data/                # Generated local database (not committed)
├── eval/                # Evaluation queries and runner
├── tests/               # Unit and integration tests (434+ tests)
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
| **Compliance** | Policy Q&A from retrieved evidence with citations | `retrieve_policy` | `agents/compliance_agent.py` |

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

### Implemented (Phases 10–12)

- **PII masking** — `gateway/pii.py` masks Aadhaar, PAN, email, phone, account numbers in agent responses and API error messages
- **Prompt-injection detection** — `gateway/input_guard.py` detects role assumption, instruction override, system prompt leak, delimiter injection, data exfiltration, SQL injection; integrated at supervisor entry point
- **Human approval gate** — `gateway/approval.py` blocks card_block, fund_transfer, account_closure, limit_increase until explicit human approval; 15-minute configurable expiration; cross-customer isolation
- **`skip_input_guard` hardened** — removed from `run_supervisor()` public API; replaced with internal `_TESTING_SKIP_INPUT_GUARD` module variable not exposed to external callers
- **API-level authentication** — FastAPI with `X-API-Key` header validation, customer_uid verification on approvals
- **Rate limiting** — not implemented (outside MVP scope)

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Python version** | 3.12.3 | CLAUDE.md targets 3.11; 3.11 unavailable in environment; all deps compatible |
| **LLM mode** | Mock by default | Zero-cost-first; `MockLLM` with `[MOCK]` prefix ensures offline development; real LLM optional via config |
| **LLM narrative** | Post-hoc explanation of tool results | LLM generates natural-language summaries from verified structured data; cannot override scores, decisions, or authorization; falls back to template on failure |
| **Intent classification** | Keyword regex, not LLM | Deterministic, fast, no API cost; fraud keywords checked before banking to avoid ambiguity |
| **Agent orchestration** | LangGraph StateGraph | Single classify → specialist → END flow; TypedDict internal state with Pydantic at entry/exit |
| **Execution model** | Sequential, not parallel | Simpler debugging; avoids rate-limit issues with free-tier LLM providers |
| **ML training** | In-memory, no persistence | Models trained from synthetic datasets each run; no `.pkl`/`.joblib` files to manage |
| **Sensitive action approval** | Blocking human-in-the-loop | `request_card_block()` creates PENDING approval; `execute_card_block()` verifies approval status with time-based expiration before execution |
| **Database** | SQLite in-memory for tests, file for dev | Fast test execution; deterministic seed (42) for reproducibility |
| **Synthetic data locale** | India (cities, INR, pin codes) | Realistic but clearly artificial; `@banksphere.test` email domain |
