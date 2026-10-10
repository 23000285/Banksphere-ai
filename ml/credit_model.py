"""Credit risk pipeline — synthetic data, gradient boosting, eligibility rules, EMI.

All data and metrics are synthetic. Do not interpret results as real-world
credit-risk performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pydantic import BaseModel, Field

SEED = 42
N_BORROWERS = 5_000
DEFAULT_RATE = 0.08


class CreditAssessmentRequest(BaseModel):
    credit_score: int = Field(..., ge=300, le=900)
    annual_income: Decimal = Field(..., gt=0)
    existing_emi_burden: Decimal = Field(default=Decimal("0"), ge=0)
    loan_amount_requested: Decimal = Field(..., gt=0)
    loan_tenure_months: int = Field(..., ge=6, le=360)
    employment_years: int = Field(default=0, ge=0)
    age: int = Field(..., ge=21, le=70)
    loan_type: str = Field(default="personal")


class CreditAssessmentResponse(BaseModel):
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_category: str
    eligible: bool
    reason_codes: list[str]
    max_eligible_amount: Decimal
    recommended_tenure_months: int
    emi: Decimal
    monthly_income: Decimal
    debt_to_income_ratio: float
    explanation_method: str = "model + deterministic rules"
    is_synthetic: bool = True


class EMIRequest(BaseModel):
    principal: Decimal = Field(..., gt=0)
    annual_rate_pct: Decimal = Field(..., ge=0, le=50)
    tenure_months: int = Field(..., ge=1, le=360)


class EMIResponse(BaseModel):
    emi: Decimal
    total_payment: Decimal
    total_interest: Decimal


def calculate_emi(request: EMIRequest) -> EMIResponse:
    p = float(request.principal)
    r = float(request.annual_rate_pct) / 100 / 12
    n = request.tenure_months

    if r > 0:
        emi_val = p * r * (1 + r) ** n / ((1 + r) ** n - 1)
    else:
        emi_val = p / n

    emi = Decimal(str(emi_val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    total = emi * n
    interest = total - request.principal

    return EMIResponse(emi=emi, total_payment=total, total_interest=interest)


def _build_synthetic_credit_dataset(
    n: int = N_BORROWERS,
    default_rate: float = DEFAULT_RATE,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    credit_scores = rng.integers(350, 850, n)
    annual_income = rng.lognormal(12.5, 0.7, n).clip(100_000, 50_000_000)
    employment_years = rng.integers(0, 35, n)
    age = rng.integers(21, 65, n)
    loan_amount = rng.lognormal(12, 1.0, n).clip(50_000, 50_000_000)
    existing_emi = rng.lognormal(8, 1.5, n).clip(0, annual_income * 0.4 / 12)
    tenure = rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], n)

    monthly_income = annual_income / 12
    dti = (existing_emi + loan_amount / tenure) / monthly_income

    score_norm = (credit_scores - 300) / 600
    income_norm = np.clip((annual_income - 100_000) / 5_000_000, 0, 1)
    emp_norm = np.clip(employment_years / 30, 0, 1)

    base_default_prob = (
        0.40 * (1 - score_norm)
        + 0.25 * np.clip(dti - 0.3, 0, 1) / 0.7
        + 0.15 * (1 - emp_norm)
        + 0.10 * (1 - income_norm)
        + 0.10 * np.where(age < 25, 0.5, 0)
    )
    base_default_prob = np.clip(base_default_prob, 0.02, 0.90)

    noise = rng.normal(0, 0.03, n)
    threshold = base_default_prob + noise
    is_default = (rng.random(n) < threshold).astype(int)

    target_defaults = int(n * default_rate)
    current_defaults = is_default.sum()
    if current_defaults > target_defaults:
        default_idx = np.where(is_default == 1)[0]
        probs = base_default_prob[default_idx]
        keep_order = np.argsort(-probs)
        flip_idx = default_idx[keep_order[target_defaults:]]
        is_default[flip_idx] = 0
    elif current_defaults < target_defaults:
        non_default_idx = np.where(is_default == 0)[0]
        probs = base_default_prob[non_default_idx]
        flip_order = np.argsort(-probs)
        flip_idx = non_default_idx[flip_order[: target_defaults - current_defaults]]
        is_default[flip_idx] = 1

    return pd.DataFrame({
        "credit_score": credit_scores,
        "annual_income": annual_income.round(2),
        "employment_years": employment_years,
        "age": age,
        "loan_amount": loan_amount.round(2),
        "existing_emi": existing_emi.round(2),
        "tenure_months": tenure,
        "dti": dti.round(4),
        "is_default": is_default,
    })


CREDIT_FEATURE_NAMES = [
    "credit_score", "annual_income", "employment_years", "age",
    "loan_amount", "existing_emi", "tenure_months", "dti",
]


@dataclass
class CreditModelArtifacts:
    model: GradientBoostingClassifier | None = None
    metrics: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=lambda: list(CREDIT_FEATURE_NAMES))


def train_credit_model(seed: int = SEED) -> CreditModelArtifacts:
    df = _build_synthetic_credit_dataset(seed=seed)
    X = df[CREDIT_FEATURE_NAMES]
    y = df["is_default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y,
    )

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return CreditModelArtifacts(
        model=model,
        metrics={
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "test_size": len(y_test),
            "test_default_count": int(y_test.sum()),
        },
    )


_INTEREST_RATES = {
    "personal": Decimal("12.0"),
    "home": Decimal("8.5"),
    "auto": Decimal("9.5"),
    "education": Decimal("10.0"),
}

_MAX_DTI = 0.50
_MIN_CREDIT_SCORE = 600


def assess_credit_risk(
    request: CreditAssessmentRequest,
    artifacts: CreditModelArtifacts,
) -> CreditAssessmentResponse:
    if artifacts.model is None:
        raise RuntimeError("Credit model not trained. Call train_credit_model() first.")

    monthly_income = Decimal(str(float(request.annual_income) / 12)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP,
    )

    emi_result = calculate_emi(EMIRequest(
        principal=request.loan_amount_requested,
        annual_rate_pct=_INTEREST_RATES.get(request.loan_type, Decimal("12.0")),
        tenure_months=request.loan_tenure_months,
    ))

    total_emi = request.existing_emi_burden + emi_result.emi
    dti = float(total_emi) / float(monthly_income) if float(monthly_income) > 0 else 999.0

    features = np.array([[
        request.credit_score,
        float(request.annual_income),
        request.employment_years,
        request.age,
        float(request.loan_amount_requested),
        float(request.existing_emi_burden),
        request.loan_tenure_months,
        dti,
    ]])

    default_prob = float(artifacts.model.predict_proba(features)[0, 1])

    reason_codes: list[str] = []
    eligible = True

    if request.credit_score < _MIN_CREDIT_SCORE:
        reason_codes.append(f"Credit score {request.credit_score} below minimum {_MIN_CREDIT_SCORE}")
        eligible = False

    if dti > _MAX_DTI:
        reason_codes.append(f"Debt-to-income ratio {dti:.2f} exceeds limit {_MAX_DTI}")
        eligible = False

    if request.employment_years < 1:
        reason_codes.append("Less than 1 year of employment")

    if default_prob > 0.3:
        reason_codes.append(f"Model default probability {default_prob:.2%} is elevated")
        eligible = False

    if not reason_codes:
        reason_codes.append("Meets all eligibility criteria")

    if default_prob >= 0.3:
        risk = "high"
    elif default_prob >= 0.1:
        risk = "medium"
    else:
        risk = "low"

    max_affordable_emi = float(monthly_income) * _MAX_DTI - float(request.existing_emi_burden)
    max_affordable_emi = max(max_affordable_emi, 0)
    rate = float(_INTEREST_RATES.get(request.loan_type, Decimal("12.0")))
    monthly_rate = rate / 100 / 12
    if monthly_rate > 0 and max_affordable_emi > 0:
        n = request.loan_tenure_months
        max_principal = max_affordable_emi * ((1 + monthly_rate) ** n - 1) / (
            monthly_rate * (1 + monthly_rate) ** n
        )
    else:
        max_principal = 0
    max_eligible = Decimal(str(round(max_principal, 2)))

    rec_tenure = request.loan_tenure_months
    if dti > _MAX_DTI and request.loan_tenure_months < 60:
        rec_tenure = min(request.loan_tenure_months + 24, 60)

    return CreditAssessmentResponse(
        default_probability=round(default_prob, 4),
        risk_category=risk,
        eligible=eligible,
        reason_codes=reason_codes,
        max_eligible_amount=max_eligible,
        recommended_tenure_months=rec_tenure,
        emi=emi_result.emi,
        monthly_income=monthly_income,
        debt_to_income_ratio=round(dti, 4),
    )
