"""Tests for credit risk pipeline — training, eligibility, EMI, reason codes."""

import warnings
from decimal import Decimal

import pytest

from ml.credit_model import (
    CREDIT_FEATURE_NAMES,
    CreditAssessmentRequest,
    CreditModelArtifacts,
    EMIRequest,
    assess_credit_risk,
    calculate_emi,
    train_credit_model,
    _build_synthetic_credit_dataset,
)

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def credit_artifacts() -> CreditModelArtifacts:
    return train_credit_model()


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class TestCreditDataset:

    def test_dataset_size(self):
        df = _build_synthetic_credit_dataset(n=1000)
        assert len(df) == 1000

    def test_default_rate_approximate(self):
        df = _build_synthetic_credit_dataset(n=5000, default_rate=0.08)
        actual = df["is_default"].mean()
        assert 0.05 <= actual <= 0.12

    def test_has_expected_columns(self):
        df = _build_synthetic_credit_dataset(n=100)
        for col in CREDIT_FEATURE_NAMES + ["is_default"]:
            assert col in df.columns

    def test_deterministic(self):
        df1 = _build_synthetic_credit_dataset(n=500, seed=42)
        df2 = _build_synthetic_credit_dataset(n=500, seed=42)
        assert df1.equals(df2)


# ---------------------------------------------------------------------------
# Model training and metrics
# ---------------------------------------------------------------------------

class TestCreditModelTraining:

    def test_model_trained(self, credit_artifacts):
        assert credit_artifacts.model is not None

    def test_roc_auc_above_threshold(self, credit_artifacts):
        assert credit_artifacts.metrics["roc_auc"] > 0.70

    def test_precision_above_threshold(self, credit_artifacts):
        assert credit_artifacts.metrics["precision"] > 0.30

    def test_recall_above_threshold(self, credit_artifacts):
        assert credit_artifacts.metrics["recall"] > 0.30

    def test_holdout_has_defaults(self, credit_artifacts):
        assert credit_artifacts.metrics["test_default_count"] > 0

    def test_deterministic(self):
        a1 = train_credit_model(seed=42)
        a2 = train_credit_model(seed=42)
        assert a1.metrics == a2.metrics


# ---------------------------------------------------------------------------
# EMI calculator
# ---------------------------------------------------------------------------

class TestEMICalculator:

    def test_basic_calculation(self):
        resp = calculate_emi(EMIRequest(
            principal=Decimal("500000"),
            annual_rate_pct=Decimal("12"),
            tenure_months=36,
        ))
        assert resp.emi > 0
        assert resp.total_payment > resp.emi
        assert resp.total_interest > 0
        assert resp.total_payment == resp.emi * 36

    def test_zero_rate(self):
        resp = calculate_emi(EMIRequest(
            principal=Decimal("120000"),
            annual_rate_pct=Decimal("0"),
            tenure_months=12,
        ))
        assert resp.emi == Decimal("10000.00")
        assert resp.total_interest == Decimal("0.00")

    def test_known_emi_value(self):
        resp = calculate_emi(EMIRequest(
            principal=Decimal("100000"),
            annual_rate_pct=Decimal("12"),
            tenure_months=12,
        ))
        assert Decimal("8800") < resp.emi < Decimal("8900")

    def test_invalid_negative_principal(self):
        with pytest.raises(Exception):
            EMIRequest(
                principal=Decimal("-100"), annual_rate_pct=Decimal("10"), tenure_months=12,
            )


# ---------------------------------------------------------------------------
# Credit assessment
# ---------------------------------------------------------------------------

class TestCreditAssessment:

    def _good_applicant(self) -> CreditAssessmentRequest:
        return CreditAssessmentRequest(
            credit_score=750,
            annual_income=Decimal("1500000"),
            existing_emi_burden=Decimal("5000"),
            loan_amount_requested=Decimal("500000"),
            loan_tenure_months=36,
            employment_years=8,
            age=35,
        )

    def _risky_applicant(self) -> CreditAssessmentRequest:
        return CreditAssessmentRequest(
            credit_score=450,
            annual_income=Decimal("200000"),
            existing_emi_burden=Decimal("8000"),
            loan_amount_requested=Decimal("1000000"),
            loan_tenure_months=12,
            employment_years=0,
            age=22,
        )

    def test_good_applicant_eligible(self, credit_artifacts):
        resp = assess_credit_risk(self._good_applicant(), credit_artifacts)
        assert resp.eligible is True
        assert resp.risk_category == "low"
        assert resp.emi > 0
        assert resp.monthly_income > 0

    def test_risky_applicant_denied(self, credit_artifacts):
        resp = assess_credit_risk(self._risky_applicant(), credit_artifacts)
        assert resp.eligible is False
        assert len(resp.reason_codes) > 0

    def test_low_credit_score_rejected(self, credit_artifacts):
        req = self._good_applicant()
        req = req.model_copy(update={"credit_score": 500})
        resp = assess_credit_risk(req, credit_artifacts)
        assert resp.eligible is False
        assert any("credit score" in r.lower() for r in resp.reason_codes)

    def test_response_has_all_fields(self, credit_artifacts):
        resp = assess_credit_risk(self._good_applicant(), credit_artifacts)
        assert resp.is_synthetic is True
        assert resp.explanation_method == "model + deterministic rules"
        assert resp.debt_to_income_ratio >= 0
        assert resp.max_eligible_amount >= 0

    def test_untrained_model_raises(self):
        empty = CreditModelArtifacts()
        with pytest.raises(RuntimeError, match="not trained"):
            assess_credit_risk(self._good_applicant(), empty)

    def test_invalid_credit_score_too_low(self):
        with pytest.raises(Exception):
            CreditAssessmentRequest(
                credit_score=200, annual_income=Decimal("500000"),
                loan_amount_requested=Decimal("100000"), loan_tenure_months=12, age=30,
            )

    def test_invalid_age_too_young(self):
        with pytest.raises(Exception):
            CreditAssessmentRequest(
                credit_score=700, annual_income=Decimal("500000"),
                loan_amount_requested=Decimal("100000"), loan_tenure_months=12, age=18,
            )
