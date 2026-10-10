"""Fraud detection pipeline — synthetic data, LightGBM classifier, Isolation Forest.

All data and metrics are synthetic. Do not interpret results as real-world
banking fraud-detection performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from pydantic import BaseModel, Field

SEED = 42
N_TRANSACTIONS = 20_000
FRAUD_RATE = 0.015

FEATURE_NAMES = [
    "amount",
    "hour",
    "is_night",
    "channel_card",
    "channel_upi",
    "channel_atm",
    "is_new_device",
    "is_foreign_city",
    "txn_velocity_1h",
    "amount_zscore",
]


class TransactionScoreRequest(BaseModel):
    amount: Decimal = Field(..., gt=0)
    hour: int = Field(..., ge=0, le=23)
    channel: str
    device_id: str | None = None
    location_city: str | None = None
    customer_avg_amount: float = Field(default=5000.0, ge=0)
    customer_std_amount: float = Field(default=3000.0, ge=0)
    customer_known_devices: list[str] = Field(default_factory=list)
    customer_home_city: str = "Mumbai"
    recent_txn_count_1h: int = Field(default=0, ge=0)


class TransactionScoreResponse(BaseModel):
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float
    risk_category: str  # "low", "medium", "high"
    top_features: list[dict[str, float]]
    explanation_method: str
    is_synthetic: bool = True


def _build_synthetic_fraud_dataset(
    n: int = N_TRANSACTIONS,
    fraud_rate: float = FRAUD_RATE,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def _make_legit(count: int) -> pd.DataFrame:
        return pd.DataFrame({
            "amount": rng.lognormal(7, 1.0, count).clip(10, 500_000),
            "hour": rng.integers(6, 23, count),
            "channel_card": rng.integers(0, 2, count),
            "channel_upi": rng.integers(0, 2, count),
            "channel_atm": rng.integers(0, 2, count),
            "is_new_device": (rng.random(count) < 0.05).astype(int),
            "is_foreign_city": (rng.random(count) < 0.03).astype(int),
            "txn_velocity_1h": rng.poisson(1.5, count).clip(0, 20),
            "amount_zscore": rng.normal(0, 0.8, count),
            "is_fraud": 0,
        })

    def _make_fraud(count: int) -> pd.DataFrame:
        return pd.DataFrame({
            "amount": rng.lognormal(9, 1.5, count).clip(100, 999_999),
            "hour": rng.choice([0, 1, 2, 3, 4, 22, 23], count),
            "channel_card": (rng.random(count) < 0.6).astype(int),
            "channel_upi": (rng.random(count) < 0.3).astype(int),
            "channel_atm": (rng.random(count) < 0.4).astype(int),
            "is_new_device": (rng.random(count) < 0.70).astype(int),
            "is_foreign_city": (rng.random(count) < 0.55).astype(int),
            "txn_velocity_1h": rng.poisson(5, count).clip(0, 20),
            "amount_zscore": rng.normal(2.5, 1.2, count),
            "is_fraud": 1,
        })

    df = pd.concat([_make_legit(n_legit), _make_fraud(n_fraud)], ignore_index=True)
    df["is_night"] = (df["hour"].isin([0, 1, 2, 3, 4, 22, 23])).astype(int)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@dataclass
class FraudModelArtifacts:
    lgb_model: lgb.Booster | None = None
    iso_forest: IsolationForest | None = None
    metrics: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))


def train_fraud_models(seed: int = SEED) -> FraudModelArtifacts:
    df = _build_synthetic_fraud_dataset(seed=seed)
    X = df[FEATURE_NAMES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y,
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "seed": seed,
        "verbose": -1,
    }
    model = lgb.train(params, train_data, num_boost_round=200)

    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    pr_auc = float(average_precision_score(y_test, y_prob))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))

    iso = IsolationForest(
        n_estimators=100, contamination=0.02, random_state=seed, n_jobs=1,
    )
    iso.fit(X_train)

    return FraudModelArtifacts(
        lgb_model=model,
        iso_forest=iso,
        metrics={
            "pr_auc": round(pr_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "test_size": len(y_test),
            "test_fraud_count": int(y_test.sum()),
            "train_size": len(y_train),
        },
        feature_names=FEATURE_NAMES,
    )


def _get_shap_explanations(
    model: lgb.Booster,
    features: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> tuple[list[dict[str, float]], str]:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        vals = shap_values.flatten()
        indices = np.argsort(np.abs(vals))[::-1][:top_k]
        top = [
            {feature_names[i]: round(float(vals[i]), 4)} for i in indices
        ]
        return top, "shap"
    except Exception:
        importances = model.feature_importance(importance_type="gain")
        indices = np.argsort(importances)[::-1][:top_k]
        top = [
            {feature_names[i]: round(float(importances[i]), 4)} for i in indices
        ]
        return top, "lightgbm_gain (SHAP unavailable)"


def score_transaction(
    request: TransactionScoreRequest,
    artifacts: FraudModelArtifacts,
) -> TransactionScoreResponse:
    if artifacts.lgb_model is None or artifacts.iso_forest is None:
        raise RuntimeError("Fraud models not trained. Call train_fraud_models() first.")

    amount = float(request.amount)
    is_new_device = 1 if (
        request.device_id and request.device_id not in request.customer_known_devices
    ) else 0
    is_foreign = 1 if (
        request.location_city and request.location_city != request.customer_home_city
    ) else 0

    std = request.customer_std_amount if request.customer_std_amount > 0 else 1.0
    zscore = (amount - request.customer_avg_amount) / std

    features = np.array([[
        amount,
        request.hour,
        1 if request.hour in (0, 1, 2, 3, 4, 22, 23) else 0,
        1 if request.channel == "card" else 0,
        1 if request.channel == "upi" else 0,
        1 if request.channel == "atm" else 0,
        is_new_device,
        is_foreign,
        request.recent_txn_count_1h,
        zscore,
    ]])

    fraud_score = float(artifacts.lgb_model.predict(features)[0])
    fraud_score = max(0.0, min(1.0, fraud_score))

    anomaly_raw = artifacts.iso_forest.decision_function(features)[0]
    anomaly_score = float(-anomaly_raw)

    if fraud_score >= 0.7:
        risk = "high"
    elif fraud_score >= 0.3:
        risk = "medium"
    else:
        risk = "low"

    top_features, method = _get_shap_explanations(
        artifacts.lgb_model, features, artifacts.feature_names,
    )

    return TransactionScoreResponse(
        fraud_score=round(fraud_score, 4),
        anomaly_score=round(anomaly_score, 4),
        risk_category=risk,
        top_features=top_features,
        explanation_method=method,
    )
