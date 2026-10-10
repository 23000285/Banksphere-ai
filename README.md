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

Phase 0 — Repository baseline established. Application implementation begins in subsequent phases.

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (after requirements.txt is populated)
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration

# Run tests
python -m pytest -q
```

## Demonstration Scenarios

1. Retrieve customer balance and recent transactions
2. Assess suspicious transactions with risk scoring
3. Estimate personal-loan eligibility and EMI
4. Answer KYC/policy questions with source citations
5. Handle card-block requests with human approval gate
6. Block prompt-injection and cross-customer data access attempts

## Disclaimer

All customers, accounts, transactions, loans, policy documents, and model metrics in this project are **synthetic**. Nothing in this system should be used for real financial decisions.
