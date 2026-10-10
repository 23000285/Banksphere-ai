"""Tests for the fraud detection pipeline — training, metrics, scoring, SHAP."""

import warnings
from decimal import Decimal

import pytest

from ml.fraud_model import (
    FEATURE_NAMES,
    FraudModelArtifacts,
    TransactionScoreRequest,
    TransactionScoreResponse,
    _build_synthetic_fraud_dataset,
    score_transaction,
    train_fraud_models,
)

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def fraud_artifacts() -> FraudModelArtifacts:
    return train_fraud_models()


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class TestFraudDataset:

    def test_dataset_size(self):
        df = _build_synthetic_fraud_dataset(n=5000)
        assert len(df) == 5000

    def test_fraud_rate_approximate(self):
        df = _build_synthetic_fraud_dataset(n=20_000, fraud_rate=0.015)
        actual_rate = df["is_fraud"].mean()
        assert 0.01 <= actual_rate <= 0.03

    def test_has_expected_columns(self):
        df = _build_synthetic_fraud_dataset(n=100)
        for col in FEATURE_NAMES + ["is_fraud"]:
            assert col in df.columns

    def test_deterministic(self):
        df1 = _build_synthetic_fraud_dataset(n=500, seed=42)
        df2 = _build_synthetic_fraud_dataset(n=500, seed=42)
        assert df1.equals(df2)

    def test_different_seed_different_data(self):
        df1 = _build_synthetic_fraud_dataset(n=500, seed=42)
        df2 = _build_synthetic_fraud_dataset(n=500, seed=99)
        assert not df1.equals(df2)


# ---------------------------------------------------------------------------
# Model training and metrics
# ---------------------------------------------------------------------------

class TestFraudModelTraining:

    def test_models_trained(self, fraud_artifacts):
        assert fraud_artifacts.lgb_model is not None
        assert fraud_artifacts.iso_forest is not None

    def test_pr_auc_above_threshold(self, fraud_artifacts):
        assert fraud_artifacts.metrics["pr_auc"] > 0.50

    def test_precision_above_threshold(self, fraud_artifacts):
        assert fraud_artifacts.metrics["precision"] > 0.50

    def test_recall_above_threshold(self, fraud_artifacts):
        assert fraud_artifacts.metrics["recall"] > 0.50

    def test_holdout_has_fraud_samples(self, fraud_artifacts):
        assert fraud_artifacts.metrics["test_fraud_count"] > 0

    def test_feature_names_preserved(self, fraud_artifacts):
        assert fraud_artifacts.feature_names == FEATURE_NAMES

    def test_deterministic_training(self):
        a1 = train_fraud_models(seed=42)
        a2 = train_fraud_models(seed=42)
        assert a1.metrics == a2.metrics


# ---------------------------------------------------------------------------
# Transaction scoring
# ---------------------------------------------------------------------------

class TestScoreTransaction:

    def _suspicious_request(self) -> TransactionScoreRequest:
        return TransactionScoreRequest(
            amount=Decimal("80000"),
            hour=2,
            channel="card",
            device_id="DEV-NEW-999",
            location_city="Chennai",
            customer_home_city="Mumbai",
            customer_known_devices=["DEV-1234"],
            recent_txn_count_1h=8,
            customer_avg_amount=3000,
            customer_std_amount=2000,
        )

    def _normal_request(self) -> TransactionScoreRequest:
        return TransactionScoreRequest(
            amount=Decimal("500"),
            hour=14,
            channel="upi",
            device_id="DEV-1234",
            location_city="Mumbai",
            customer_home_city="Mumbai",
            customer_known_devices=["DEV-1234"],
            recent_txn_count_1h=1,
            customer_avg_amount=3000,
            customer_std_amount=2000,
        )

    def test_suspicious_gets_high_score(self, fraud_artifacts):
        resp = score_transaction(self._suspicious_request(), fraud_artifacts)
        assert resp.fraud_score >= 0.5
        assert resp.risk_category in ("medium", "high")

    def test_normal_gets_low_score(self, fraud_artifacts):
        resp = score_transaction(self._normal_request(), fraud_artifacts)
        assert resp.fraud_score < 0.5
        assert resp.risk_category == "low"

    def test_response_has_all_fields(self, fraud_artifacts):
        resp = score_transaction(self._normal_request(), fraud_artifacts)
        assert isinstance(resp, TransactionScoreResponse)
        assert 0 <= resp.fraud_score <= 1
        assert resp.risk_category in ("low", "medium", "high")
        assert len(resp.top_features) > 0
        assert resp.is_synthetic is True

    def test_shap_or_fallback_explanation(self, fraud_artifacts):
        resp = score_transaction(self._normal_request(), fraud_artifacts)
        assert resp.explanation_method in ("shap", "lightgbm_gain (SHAP unavailable)")

    def test_untrained_model_raises(self):
        empty = FraudModelArtifacts()
        with pytest.raises(RuntimeError, match="not trained"):
            score_transaction(
                TransactionScoreRequest(amount=Decimal("100"), hour=12, channel="upi"),
                empty,
            )

    def test_invalid_amount_rejected(self):
        with pytest.raises(Exception):
            TransactionScoreRequest(amount=Decimal("-100"), hour=12, channel="upi")

    def test_invalid_hour_rejected(self):
        with pytest.raises(Exception):
            TransactionScoreRequest(amount=Decimal("100"), hour=25, channel="upi")
