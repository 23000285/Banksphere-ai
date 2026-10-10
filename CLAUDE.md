# Multi-Agent Banking MVP — Claude Code Project Guide

> **Purpose:** Build the simulated real-time banking multi-agent MVP in controlled, verifiable phases, with clear Git history.  
> **Rule:** This file is the shared project guide. It is **not** permission to implement every phase in one go. Work only on the phase explicitly requested by the user, then stop for review.

## 1. Project goal

Build a demonstrable banking MVP with a supervisor agent and specialist agents for banking, fraud/risk, credit/loans, and compliance/RAG. Use synthetic data only. This is a proof of concept, not a real banking system.

### Main demonstration scenarios

1. Retrieve a selected customer's balance and last five transactions.
2. Assess a suspicious transaction and explain its risk score.
3. Estimate personal-loan eligibility and EMI.
4. Answer a KYC/policy question using retrieved policy evidence and source citations.
5. Handle a card-block request safely, requiring explicit human approval before performing the simulated sensitive action.
6. Block or safely reject an attempt to bypass instructions or access another customer's data.

## 2. Target stack

Use these choices unless the repository/environment or a documented blocker justifies a minimal change:

- **Language:** Python 3.11
- **Agent orchestration:** LangGraph `StateGraph`; a small explicit Python fallback is acceptable if LangGraph becomes a blocker, but document the trade-off
- **LLM interface:** LiteLLM, with model/provider configured centrally
- **Structured data and validation:** Pydantic v2
- **API:** FastAPI + Uvicorn
- **Database:** SQLite + SQLAlchemy
- **Synthetic data:** Faker + NumPy + pandas
- **Fraud ML:** LightGBM (or XGBoost if needed) plus Isolation Forest for anomaly detection
- **Credit risk ML:** scikit-learn or gradient boosting; pair the model with deterministic eligibility rules and an EMI calculator
- **Model explanation:** SHAP where practical; otherwise expose truthful, clearly labelled model-specific feature explanations and document the limitation
- **RAG:** ChromaDB + BM25 (`rank_bm25`); small sentence-transformer embedding model; reranker is optional
- **Guardrails:** input validation, PII masking, prompt-injection checks, per-agent tool permissions, approval gate, audit log
- **UI:** Streamlit; do not build a separate React UI for this MVP
- **Tests:** pytest and a small repeatable evaluation set
- **Artifacts:** `README.md`, `ARCHITECTURE.md`, `requirements.txt`, `.env.example`, `.gitignore`

Prefer a low-resource implementation: load models once, avoid unnecessary parallel LLM calls, cache only safe synthetic-data results when useful, and offer a deterministic mock-LLM mode for tests/demo resilience.

## 3. Required architecture boundaries

Suggested structure (adapt only if an existing structure is already established):

```text
banking-agents/
├── app/                 # FastAPI entry point and API schemas
├── ui/                  # Streamlit application
├── agents/              # supervisor, banking, fraud, credit, compliance
├── tools/               # registry and domain tools
├── ml/                  # data generation, training, model artifacts
├── rag/                 # synthetic policy documents, ingestion, retrieval
├── gateway/             # PII masking, input checks, routing/config, audit
├── data/                # generated local database; do not commit secrets
├── eval/                # evaluation queries and runner
├── tests/
├── README.md
├── ARCHITECTURE.md
├── CLAUDE.md
├── requirements.txt
├── .env.example
└── .gitignore
```

### Agent responsibilities

- **Supervisor:** classify intent, produce a validated routing plan, select specialists, combine their results, and escalate uncertain/high-risk actions. It should not directly access the database or bypass tools.
- **Banking agent:** customer profile, accounts, transactions, and card status through authorized tools.
- **Fraud agent:** transaction score, anomaly signal, explanation, and a risk recommendation such as allow / verify / block. A model suggestion alone must not perform a sensitive action.
- **Credit agent:** credit profile, default-risk estimate, eligibility rules, and EMI calculation.
- **Compliance agent:** answer only from retrieved policy evidence, cite source document names, and decline unsupported questions.
- **Notification stub:** simulate notification by writing a structured log/message; do not send real messages.

### Security and correctness requirements

- Use synthetic customers, accounts, transactions, loans, and policy documents only.
- Never commit `.env`, API keys, tokens, passwords, real personal data, or private database files.
- `.env.example` contains placeholder values only.
- Enforce customer/tenant authorization in tools, not just in the prompt or UI.
- Each agent may call only its allowed tools. Validate tool arguments with Pydantic.
- Log tool identity, request/trace ID, outcome, and relevant timing without logging secrets or unnecessary PII.
- Do not allow an LLM response alone to authorize or execute a money-moving or sensitive action. Require an explicit human approval step for simulated card-block/transfer actions.
- Treat user prompts and retrieved documents as untrusted input. Test prompt-injection attempts and cross-customer data access.
- The ML data and metrics are synthetic. Never describe them as real-world banking performance.
- The RAG agent must not invent a citation or claim a source supports content that was not retrieved.
- If a test or safety check fails, do not move to the next phase.

## 4. How Claude Code must work

For **every requested phase**, follow this workflow:

1. **Inspect before editing.** Check the relevant files, Python/environment version, dependency setup, Git branch/status, and available tests. Do not assume the repository state.
2. **Plan narrowly.** State the phase objective, intended files, prerequisites, and verification commands in a few lines.
3. **Implement only that phase.** Preserve unrelated user changes and working behavior. Do not implement later phases or add unnecessary dependencies.
4. **Test the change.** Run relevant unit tests, lint/compile checks if configured, and a practical smoke test. Report exact commands and their actual outputs/results. Do not claim a check passed if it was not run.
5. **Review the diff.** Check for unintended files, secrets, `.env`, generated datasets/model binaries, database artifacts, whitespace errors, and unrelated refactors. Run `git diff --check` when applicable.
6. **Report and stop.** Summarize files changed, behavior implemented, tests passed/failed, known limitations, manual verification steps, and the proposed commit message.
7. **Wait for approval.** Do not automatically commit, push, start the next phase, or claim the phase is complete on the user's behalf. The user reviews the changes and performs the Git checkpoint.

If blocked, report the smallest blocker and ask for the needed choice or input. Do not silently replace the required architecture. Never invent a GitHub remote URL. If the remote is missing, ask the user to supply/confirm it.

### Git checkpoint protocol

After Claude Code stops, the user should inspect the state and stage only intended files:

```powershell
git status
git diff --check
git diff
```

After reviewing, stage the specific files for that phase (prefer explicit paths over blindly staging everything), then inspect the staged diff:

```powershell
git add <intended-paths>
git diff --cached
git status
```

Only after verifying that no credentials, `.env`, private data, or unintended artifacts are staged:

```powershell
git commit -m "<phase commit message>"
git push
```

If a push fails because the branch has no upstream, inspect the branch and remote and ask the user before selecting a push command. Never force-push unless the user explicitly requests it and understands the impact.

## 5. Today's eight-hour schedule — Day 1

**Target:** establish the repository and deliver a terminal-tested backend with synthetic data, banking tools, fraud/credit tools, and supervisor routing. Time estimates are targets; correctness takes priority over the clock.

| Time | Phase | Work | Completion gate | Suggested commit |
|---|---|---|---|---|
| 0:00–0:30 | 0. Repository baseline | Check local folder, Python 3.11, Git status/branch/remotes; ensure the new repo is connected or identify missing URL; create minimal `.gitignore`; record a baseline without destroying any existing work. | Cleanly understood repository state; no secrets staged; remote status known. | `chore: establish repository baseline` |
| 0:30–1:00 | 1. Scaffold and configuration | Create package structure, `requirements.txt`, `.env.example`, minimal `README.md`/`ARCHITECTURE.md`, centralized LLM config and mock mode; perform import/startup checks. | Project imports/starts; config is externalized; `.env` is ignored. | `chore: scaffold banking agent MVP` |
| 1:00–2:00 | 2. Synthetic banking data | Create SQLAlchemy schema and Faker/NumPy generator for customers, accounts, transactions and loans; target about 200 customers and 50,000 transactions if feasible. | Database/schema and relationships validate; row counts and sample queries pass; seed reruns are controlled. | `feat: generate synthetic banking data` |
| 2:00–3:00 | 3. Banking tools and registry | Build customer profile, accounts, transactions and card-status tools; add Pydantic schemas, permission map, customer-level access checks, and audit events. | Unit tests pass for valid/invalid inputs and cross-customer access denial. | `feat: add banking tools and permissioned registry` |
| 3:00–5:00 | 4. Fraud and credit ML tools | Generate labelled synthetic data; train/evaluate a fraud classifier and Isolation Forest; report precision, recall and PR-AUC; train a credit-risk model; add explanations and EMI/eligibility functions; save/load model artifacts only if the pipeline needs them. | Reproducible holdout metrics; both ML tool interfaces work; explanations and limitations are honest. | `feat: add fraud and credit risk models` |
| 5:00–6:00 | 5. Banking and fraud agents | Implement controlled agent loops with max-step/error handling; give each agent only its allowed tools; connect banking and fraud use cases. | Balance/history and suspicious-transaction scenarios work from a Python script. | `feat: implement banking and fraud agents` |
| 6:00–7:00 | 6. Credit agent and supervisor | Implement credit agent and LangGraph supervisor with Pydantic-validated routing state; support clear routing failure/uncertainty handling. | Banking, fraud, and credit requests route to the expected specialist in a script/test. | `feat: add credit agent and supervisor routing` |
| 7:00–8:00 | 7. Day-1 integration and regression | Run all current CLI use cases; fix only blocking integration/routing issues; add regression tests; document known gaps and prepare a clean diff. | End-to-end terminal smoke tests pass; failing checks are recorded honestly; no secrets/unintended artifacts are staged. | `test: verify day one banking agent workflow` |

**Important:** If training or debugging takes longer, do not fake metrics or skip authorization tests. First simplify optional details (such as sophisticated SHAP visualization or model tuning), preserve a working fraud pipeline and the supervisor, record the compromise in `ARCHITECTURE.md`, and ask the user before crossing the time/scope boundary.

## 6. Day 2 roadmap — separate future phases

Do not start these phases during Day 1 unless the user explicitly asks. Day 2 is another eight-hour block.

| Time | Phase | Work | Completion gate | Suggested commit |
|---|---|---|---|---|
| 0:00–1:00 | 8. Policy corpus and hybrid RAG | Write 5–6 short synthetic KYC/loan/fraud/card policy files; implement ingestion, ChromaDB vector retrieval and BM25 result merging. | Retrieval tests return relevant chunks with source names. | `feat: add hybrid banking policy retrieval` |
| 1:00–2:00 | 9. Compliance agent and notification stub | Add evidence-grounded policy answers with citations/refusal when unsupported; log simulated notifications. | KYC question returns sources; unknown question does not invent an answer. | `feat: add compliance agent and notification stub` |
| 2:00–3:00 | 10. Gateway and approval controls | Add PII masking, prompt-injection checks, audit log, amount/action limits and human approval pause for sensitive actions. | Injection and cross-customer tests are denied; sensitive actions cannot complete without approval. | `feat: add gateway guardrails and approval audit` |
| 3:00–3:45 | 11. FastAPI | Expose workflow through FastAPI with Pydantic request validation, basic API-key/JWT check, trace IDs and safe errors. | Swagger loads; valid/invalid API smoke tests pass. | `feat: expose agents through FastAPI` |
| 3:45–5:00 | 12. Streamlit UI | Add customer selector, chat, expandable agent/tool trace, approval interaction and a risk visualization. | Core scenarios are usable in a browser. | `feat: build Streamlit chat and agent trace UI` |
| 5:00–6:00 | 13. Evaluation | Create 20–30 evaluation queries; measure routing accuracy, fraud PR-AUC, and simple RAG groundedness/refusal behavior; fix top failures. | Evaluation runner produces repeatable documented results. | `test: add routing fraud and RAG evaluations` |
| 6:00–8:00 | 14. Submission and demo | Finalize README, architecture diagram/notes, limitations, setup instructions, sample prompts and demo script; test a clean setup; record a 5–10 minute demo and check final GitHub status. | A reviewer can run the app from documented steps; required links/artifacts are ready. | `docs: finalize setup architecture and demo guide` |

## 7. Copy one prompt at a time

Paste **only one** phase prompt into Claude Code for each work session. The global workflow and safety requirements in this file apply to each prompt. Once Claude Code reports results, review them before committing and asking for the next phase.

### Phase 0 prompt — Repository baseline

```text
Work on Phase 0 only: Repository baseline (target 30 minutes).

Inspect the current workspace, Python version, Git status, current branch, remotes, and ignored/untracked files. The GitHub repository is newly created and may not contain application code. Do not assume a remote URL and do not overwrite existing files. Confirm whether Python 3.11 is available and whether the local repository is connected to the intended GitHub remote. Check for likely secret files and make sure .env and local sensitive artifacts are ignored. Create only the minimal baseline files needed for this phase, if missing.

Run appropriate checks and report exact results. Summarize the repository state, files changed, missing inputs, manual verification steps, and proposed commit message: "chore: establish repository baseline".

STOP after Phase 0. Do not scaffold the application, implement later phases, commit, or push. Wait for my review.
```

### Phase 1 prompt — Scaffold and configuration

```text
Work on Phase 1 only: scaffold and configuration (target 30 minutes). First inspect Phase 0 results and the current Git diff. Create the minimal Python 3.11 project structure from CLAUDE.md, requirements.txt, .env.example, a proper .gitignore, and starter README.md and ARCHITECTURE.md. Add centralized LiteLLM/provider configuration without embedding credentials, plus a deterministic mock mode suitable for tests when no API key is configured. Do not build agents or ML pipelines yet.

Run dependency/import/startup checks that are possible in this environment and check that .env is ignored. Report commands/results, changed files, limitations, and proposed commit message: "chore: scaffold banking agent MVP". STOP for review; do not commit, push, or begin Phase 2.
```

### Phase 2 prompt — Synthetic data

```text
Work on Phase 2 only: synthetic banking data (target 60 minutes). Inspect the existing scaffold first. Implement SQLAlchemy models and a repeatable Faker/NumPy generator for synthetic customers, accounts, transactions, and loans. Aim for about 200 customers and 50,000 transactions where the environment allows it. Include consistent foreign keys, valid timestamps/amounts, realistic but artificial transaction fields, and a controlled seed/reset approach. Do not use real personal or financial data.

Add tests or a verification script for row counts, relationships, sample queries, duplicate prevention/reseed behavior, and database creation. Run them and report exact outcomes. Suggested commit: "feat: generate synthetic banking data". STOP for review; do not start Phase 3.
```

### Phase 3 prompt — Banking tools and registry

```text
Work on Phase 3 only: banking tools and permissioned registry (target 60 minutes). Inspect the schema and current tests. Implement tools for customer profile, accounts, transactions, and card status. Define Pydantic input/output schemas, register tools centrally, enforce the permitted tool list and customer-level authorization, and produce structured audit events without exposing secrets or unnecessary PII. Do not implement LLM agents yet.

Add tests for normal requests, malformed inputs, missing customers, and attempts to read another customer's records. Run all relevant tests and report exact results. Suggested commit: "feat: add banking tools and permissioned registry". STOP for review; do not start Phase 4.
```

### Phase 4 prompt — Fraud and credit ML

```text
Work on Phase 4 only: fraud and credit-risk models (target 120 minutes). Inspect existing schemas, dependencies, and available compute first. Build a reproducible synthetic fraud dataset with roughly 1–2% fraud if practical, using injected patterns such as unusual amounts, odd hours, new devices/locations, and transaction velocity. Train a fraud classifier (LightGBM; use XGBoost or an appropriate scikit-learn fallback only if necessary) and an Isolation Forest anomaly detector. Evaluate on a holdout set and report precision, recall, and PR-AUC; do not rely on accuracy alone. Add a score_transaction interface returning score, recommended risk category, and top explanatory features (SHAP where practical; accurately label any fallback).

Build a separate synthetic credit-risk pipeline using borrower/loan features. Add prediction and reason codes plus deterministic eligibility checks and an EMI calculator. Keep training and inference deterministic where practical, validate inputs, and do not claim synthetic metrics reflect real-world performance. Avoid committing large generated datasets or model binaries unless the project genuinely requires tracked artifacts; document how to regenerate them.

Run tests and report real metric values, exact commands, changed files, and limitations. Suggested commit: "feat: add fraud and credit risk models". STOP for review; do not start Phase 5.
```

### Phase 5 prompt — Banking and fraud agents

```text
Work on Phase 5 only: Banking and Fraud agents (target 60 minutes). Inspect the registry, schemas, and ML tool interfaces. Implement a banking agent and fraud agent that call only their authorized tools, use validated inputs/outputs, limit steps, handle tool/model errors safely, and return traceable structured results. The fraud agent may recommend allow/verify/block but must not bypass approval for a sensitive action. Do not implement the credit agent or supervisor yet.

Add script/tests for balance plus last-five-transactions and suspicious-transaction scoring. Run relevant tests and report the actual outcomes and manual checks. Suggested commit: "feat: implement banking and fraud agents". STOP for review; do not start Phase 6.
```

### Phase 6 prompt — Credit agent and supervisor

```text
Work on Phase 6 only: Credit agent and LangGraph supervisor (target 60 minutes). Inspect the current agent/tool interfaces. Implement the credit agent using existing credit-risk and EMI tools, then add a supervisor graph with Pydantic-validated routing state. Route banking, fraud, and credit requests to the expected specialist, handle ambiguous/unsupported requests safely, enforce a bounded workflow, and aggregate specialist results without exposing internal secrets. Keep the execution sequential if free-tier rate limits or resource constraints make parallel calls unreliable. Do not start RAG, API, or UI work.

Add tests for expected routing, malformed plans, tool failure, and the key use cases. Run them and report actual results. Suggested commit: "feat: add credit agent and supervisor routing". STOP for review; do not start Phase 7.
```

### Phase 7 prompt — Day-1 integration verification

```text
Work on Phase 7 only: Day-1 CLI integration and regression (target 60 minutes). Run the current project end to end from the terminal. Exercise customer balance/history, suspicious transaction scoring/explanation, credit-risk/eligibility/EMI, and combined routing where currently implemented. Fix only blocking defects in the completed Day-1 scope; add regression tests for fixes. Do not start RAG, gateway, FastAPI, Streamlit, or other Day-2 scope.

Review all changes for secrets, accidental data/model artifacts, authorization bypasses, and unrelated edits. Run tests and git diff --check, reporting exact results and any unresolved failures. Suggested commit: "test: verify day one banking agent workflow". STOP for review.
```

### Phase 8 prompt — Policy corpus and hybrid RAG

```text
Work on Phase 8 only: policy corpus and hybrid retrieval (target 60 minutes). Create 5–6 short, clearly synthetic policy documents for KYC, minor accounts, loans, fraud disputes, and card rules. Implement ingestion into ChromaDB and BM25 retrieval, then merge results with source metadata preserved. Keep the corpus small and make ingestion repeatable. Do not implement the compliance agent yet.

Test relevant retrieval, source names, empty results, and repeat ingestion. Report actual outcomes and suggested commit: "feat: add hybrid banking policy retrieval". STOP for review; do not start Phase 9.
```

### Phase 9 prompt — Compliance agent and notification stub

```text
Work on Phase 9 only: Compliance agent and notification stub (target 60 minutes). Use the Phase 8 retriever to answer policy questions only from retrieved evidence. Include truthful source-document citations, a clear refusal/clarification when evidence is absent, and tests against unsupported claims. Add a notification stub that writes a structured simulated event; it must not send real email/SMS. Do not start gateway or UI work.

Run relevant tests and report exact results. Suggested commit: "feat: add compliance agent and notification stub". STOP for review; do not start Phase 10.
```

### Phase 10 prompt — Gateway, guardrails and approval

```text
Work on Phase 10 only: gateway, guardrails and approval controls (target 60 minutes). Inspect existing agent/tool boundaries first. Add input validation, practical PII masking, prompt-injection checks, per-agent tool permissions, structured audit logging, safe error handling, and a human approval state for simulated card-block/transfer actions. A model or agent recommendation must never silently approve its own sensitive action. Avoid logging secrets or unnecessary PII.

Add adversarial tests for prompt injection, cross-customer data requests, malformed inputs, and sensitive actions without approval. Run them and report results and limitations. Suggested commit: "feat: add gateway guardrails and approval audit". STOP for review; do not start Phase 11.
```

### Phase 11 prompt — FastAPI

```text
Work on Phase 11 only: FastAPI API (target 45 minutes). Expose the current workflow using Pydantic request/response schemas, a request/trace ID, appropriate validation and safe errors, and a basic API-key or JWT check suitable for an MVP. Do not store real secrets in code; provide placeholder configuration only. Do not start Streamlit work.

Run API tests and smoke checks, verify Swagger/OpenAPI is accessible when the server is started, and report the exact commands and results. Suggested commit: "feat: expose agents through FastAPI". STOP for review; do not start Phase 12.
```

### Phase 12 prompt — Streamlit UI

```text
Work on Phase 12 only: Streamlit banking interface (target 75 minutes). Build a simple customer selector and chat interface, an expandable agent/tool trace showing safe summaries of agent, tool, timing and outcome, a human-approval interaction, and one risk-score visualization. Prefer calling the existing workflow/API rather than duplicating agent logic. Do not build a separate React frontend.

Run available tests and a UI startup/smoke check; document manual browser verification steps. Suggested commit: "feat: build Streamlit chat and agent trace UI". STOP for review; do not start Phase 13.
```

### Phase 13 prompt — Evaluation

```text
Work on Phase 13 only: evaluation and regression report (target 60 minutes). Create a repeatable set of 20–30 synthetic queries with expected specialist routes and representative banking, fraud, credit, and compliance cases. Measure routing accuracy, report the fraud model's holdout PR-AUC and other important metrics, and implement a simple RAG groundedness/refusal check. Clearly separate automated metrics from manual judgment. Do not invent or hardcode results as if they were measured.

Run the evaluation, save a concise results artifact suitable for the README, fix only the most important failures within this phase, and rerun relevant tests. Suggested commit: "test: add routing fraud and RAG evaluations". STOP for review; do not start Phase 14.
```

### Phase 14 prompt — Documentation, demo and final review

```text
Work on Phase 14 only: final documentation and demo readiness (target 120 minutes). Complete README.md with prerequisites, environment setup, safe secret handling, install/run/test commands, sample prompts, measured evaluation results, known limitations, and the synthetic-data disclaimer. Complete ARCHITECTURE.md with a diagram or clear workflow, agent responsibilities, tool permissions, safety/approval flow, design decisions, trade-offs and future work. Verify .env.example contains placeholders only, and .gitignore excludes secrets and local artifacts.

Test the documented setup in a clean environment/folder if feasible. Prepare a 5–10 minute demo script covering architecture, synthetic data/metrics, the main use cases, agent traces, a prompt-injection attempt, approval gate, evaluation results and limitations. Do not claim a demo was recorded or uploaded unless that actually happened. Perform a final test/status/secret review and report what remains for the user to do. Suggested commit: "docs: finalize setup architecture and demo guide". STOP for user review; do not commit, push, or submit automatically.
```

## 8. Suggested local run commands

Use the commands appropriate to the final structure and platform; verify the actual filenames before running.

```powershell
# Create and activate a virtual environment (Windows PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt

# Tests
python -m pytest -q

# Typical app commands after those entry points have been implemented
uvicorn app.main:app --reload
streamlit run ui/streamlit_app.py
```

Do not run every command blindly: confirm that the target file/entry point exists in the current phase, and run the relevant command for the work completed so far.

## 9. Completion checklist for every phase

- [ ] Claude Code inspected relevant existing files and Git state.
- [ ] Only the requested phase was implemented.
- [ ] Relevant tests/smoke checks were executed and results are truthful.
- [ ] Authorization, synthetic-data, and secrets requirements remain satisfied.
- [ ] Diff and staged files were reviewed; no unrelated changes/secrets included.
- [ ] Claude Code stopped and supplied the proposed commit message.
- [ ] User reviewed and committed/pushed the phase before requesting the next phase.

## 10. Final delivery checklist

- [ ] GitHub repository has readable, phase-based commits.
- [ ] `.env`/credentials are not tracked; `.env.example` contains placeholders.
- [ ] Synthetic data is clearly disclosed in code and documentation.
- [ ] Supervisor and specialist agents work for 3–5 main use cases.
- [ ] Banking tools enforce customer-level authorization.
- [ ] Fraud metrics include PR-AUC, precision and recall.
- [ ] Compliance answers include evidence-backed source names and abstain when evidence is missing.
- [ ] Sensitive actions pause for human approval and are audited.
- [ ] API/UI work and evaluation are verified.
- [ ] README, architecture documentation, demo script/video link, limitations, and run commands are accurate.
