# BankSphere AI: Multi-Agent Intelligent Banking Automation System

A simulated real-time banking application powered by multi-agent AI. This is a **proof-of-concept MVP** using entirely **synthetic data** — it does not connect to real banking systems or use real personal/financial information.

## Overview

BankSphere AI demonstrates a supervisor-agent architecture where specialist agents handle banking operations, fraud detection, credit risk assessment, and compliance queries. All data, metrics, and model outputs are synthetic and must not be interpreted as real-world banking performance.

## Target Stack

- **Language:** Python 3.12
- **Agent orchestration:** LangGraph StateGraph
- **LLM interface:** LiteLLM (centrally configured)
- **Validation:** Pydantic v2
- **API:** FastAPI + Uvicorn
- **Database:** SQLite + SQLAlchemy
- **Synthetic data:** Faker + NumPy + pandas
- **Fraud ML:** LightGBM + Isolation Forest
- **Credit ML:** scikit-learn / gradient boosting
- **RAG:** ChromaDB + BM25 + sentence-transformers
- **UI:** Streamlit
- **Tests:** pytest

## Project Status

All 14 phases are implemented:

- Synthetic banking data (200 customers, ~50K transactions, loans)
- Banking tools with customer-level authorization and audit logging
- Fraud detection (LightGBM + Isolation Forest + SHAP explanations)
- Credit risk assessment with eligibility rules and EMI calculator
- Hybrid RAG (ChromaDB + BM25) for 6 synthetic policy documents
- Compliance agent with evidence-grounded citations and refusal
- LangGraph supervisor routing (banking, fraud, credit, compliance, card_block)
- Gateway guardrails: PII masking, prompt-injection detection, input validation
- Human approval gate for sensitive actions with time-based expiration
- FastAPI REST API with API-key auth and trace IDs
- Streamlit UI with Banking Dashboard, AI Assistant, and Risk & Approvals tabs
- 434+ automated tests

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template — defaults to mock mode (no API keys needed)
cp .env.example .env

# Run tests
python -m pytest -q

# Start the Streamlit UI
streamlit run ui/streamlit_app.py

# Or start the FastAPI server
uvicorn app.main:app --reload
```

## LLM Provider Configuration

The application defaults to **mock mode** — no API keys, no network access, no cost. All mock responses are prefixed with `[MOCK]` so they are never mistaken for real LLM output.

| Provider | Cost | Setup |
|---|---|---|
| `mock` (default) | Free | No configuration needed |
| `gemini` | Free tier (limited quota) | Set `GEMINI_API_KEY` in `.env` |
| `bedrock` | Paid per token | Configure AWS credentials and region |

To switch providers, edit `.env`:

```bash
# Option 1: Mock (default) — no keys needed
LLM_PROVIDER=mock
MOCK_LLM=true

# Option 2: Gemini free tier — requires API key
LLM_PROVIDER=gemini
LLM_MODEL=gemini/gemini-3.8-flash
MOCK_LLM=false
GEMINI_API_KEY=your-key-here

# Option 3: Bedrock — requires AWS credentials and incurs charges
LLM_PROVIDER=bedrock
LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
MOCK_LLM=false
AWS_REGION=us-east-1
```

**Important:** Never commit API keys. The `.env` file is gitignored. Free-tier quotas (Gemini) are limited and not guaranteed — do not rely on them for automated tests.

### LLM Integration

When Gemini (or another provider) is enabled, the agents use the LLM to generate natural-language narratives that explain their verified tool results. The deterministic core is unchanged:

- **Intent classification** remains keyword/regex-based (no LLM)
- **Banking agent** queries the database directly; LLM summarizes the results
- **Fraud agent** scores with LightGBM/Isolation Forest; LLM explains the scores
- **Credit agent** uses business rules + gradient boosting; LLM explains eligibility
- **Compliance agent** retrieves policy via hybrid RAG; LLM summarizes with citations
- **Card block** follows the approval gate; LLM describes the status

The LLM cannot override fraud scores, credit decisions, eligibility rules, or approval gates. If the LLM call fails, the system falls back to the structured/template response with no loss of functionality.

## Demonstration Scenarios

1. Retrieve customer balance and recent transactions
2. Assess suspicious transactions with risk scoring
3. Estimate personal-loan eligibility and EMI
4. Answer KYC/policy questions with source citations
5. Handle card-block requests with human approval gate
6. Block prompt-injection and cross-customer data access attempts

## Disclaimer

All customers, accounts, transactions, loans, policy documents, and model metrics in this project are **synthetic**. Nothing in this system should be used for real financial decisions.
