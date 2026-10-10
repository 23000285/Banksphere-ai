# BankSphere AI — Architecture

## System Overview

BankSphere AI uses a supervisor-agent pattern where a central supervisor classifies user intent and routes requests to specialist agents, each with restricted tool access and validated inputs/outputs.

## Directory Structure

```
Banksphere-ai/
├── app/                 # FastAPI entry point and API schemas
├── ui/                  # Streamlit application
├── agents/              # Supervisor, banking, fraud, credit, compliance agents
├── tools/               # Tool registry and domain-specific tools
├── ml/                  # Data generation, training, model artifacts
├── rag/                 # Synthetic policy documents, ingestion, retrieval
├── gateway/             # PII masking, input checks, routing config, audit
├── data/                # Generated local database (not committed)
├── eval/                # Evaluation queries and runner
├── tests/               # Unit and integration tests
├── CLAUDE.md            # Project guide for Claude Code
├── README.md            # Project documentation
├── ARCHITECTURE.md      # This file
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template (no real secrets)
└── .gitignore           # Excludes secrets, databases, model artifacts
```

## Agent Responsibilities

| Agent | Role | Allowed Tools |
|---|---|---|
| **Supervisor** | Intent classification, routing, result aggregation | Routing only — no direct DB or tool access |
| **Banking** | Customer profiles, accounts, transactions, card status | Banking query tools |
| **Fraud** | Transaction scoring, anomaly detection, risk recommendation | Fraud ML tools |
| **Credit** | Credit risk, eligibility rules, EMI calculation | Credit ML tools |
| **Compliance** | Policy Q&A from retrieved evidence with citations | RAG retrieval tools |

## Security Boundaries

- Customer-level authorization enforced in tools, not just prompts
- Per-agent tool permission enforcement
- Human approval gate for sensitive actions (card block, transfers)
- PII masking and prompt-injection checks at the gateway
- Structured audit logging without secrets or unnecessary PII
- All data is synthetic — clearly disclosed in code and documentation

## Design Decisions

*To be expanded as implementation progresses through each phase.*
