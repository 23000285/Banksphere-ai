"""Credit agent — credit risk assessment, eligibility, and EMI calculation.

Calls only tools in the 'credit' permission set via the central registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from agents.base import AgentResult, BaseAgent
from gateway.audit import AuditTimer
from ml.credit_model import (
    CreditAssessmentRequest,
    CreditModelArtifacts,
    EMIRequest,
    assess_credit_risk,
    calculate_emi,
)
from tools.banking_tools import AuthorizationDenied, CustomerNotFound
from tools.registry import ToolPermissionDenied, invoke_tool
from tools.schemas import CustomerProfileRequest

AGENT_NAME = "credit"


@dataclass
class CreditAgent(BaseAgent):
    name: str = AGENT_NAME

    def assess_loan_eligibility(
        self,
        session: Session,
        customer_uid: str,
        credit_artifacts: CreditModelArtifacts,
        loan_amount: Decimal,
        loan_tenure_months: int = 36,
        loan_type: str = "personal",
        credit_score: int = 700,
        annual_income: Decimal = Decimal("600000"),
        existing_emi_burden: Decimal = Decimal("0"),
        employment_years: int = 3,
        age: int = 30,
    ) -> AgentResult:
        profile = self._call_tool(
            "get_customer_profile",
            session=session,
            request=CustomerProfileRequest(
                customer_uid=customer_uid,
                requesting_customer_uid=customer_uid,
            ),
            agent=AGENT_NAME,
        )
        if profile is None:
            return self._make_result("error", error=self._steps[-1].result_summary)

        limit_hit = self._check_step_limit()
        if limit_hit:
            return limit_hit

        with AuditTimer() as timer:
            try:
                request = CreditAssessmentRequest(
                    credit_score=credit_score,
                    annual_income=annual_income,
                    existing_emi_burden=existing_emi_burden,
                    loan_amount_requested=loan_amount,
                    loan_tenure_months=loan_tenure_months,
                    employment_years=employment_years,
                    age=age,
                    loan_type=loan_type,
                )
                assessment = assess_credit_risk(request, credit_artifacts)
            except RuntimeError as exc:
                self._record_step("assess_credit_risk", "error", str(exc), timer.elapsed_ms)
                return self._make_result("error", error=str(exc))
            except Exception as exc:
                self._record_step("assess_credit_risk", "error", str(exc), timer.elapsed_ms)
                return self._make_result("error", error=str(exc))

        self._record_step(
            "assess_credit_risk",
            "success",
            f"eligible={assessment.eligible} risk={assessment.risk_category}",
            timer.elapsed_ms,
        )

        return self._make_result(
            "completed",
            response={
                "customer_uid": customer_uid,
                "customer_name": f"{profile.first_name} {profile.last_name}",
                "eligible": assessment.eligible,
                "risk_category": assessment.risk_category,
                "default_probability": assessment.default_probability,
                "reason_codes": assessment.reason_codes,
                "emi": str(assessment.emi),
                "monthly_income": str(assessment.monthly_income),
                "debt_to_income_ratio": assessment.debt_to_income_ratio,
                "max_eligible_amount": str(assessment.max_eligible_amount),
                "recommended_tenure_months": assessment.recommended_tenure_months,
                "is_synthetic": assessment.is_synthetic,
            },
        )

    def calculate_emi_only(
        self,
        principal: Decimal,
        annual_rate_pct: Decimal,
        tenure_months: int,
    ) -> AgentResult:
        limit_hit = self._check_step_limit()
        if limit_hit:
            return limit_hit

        with AuditTimer() as timer:
            try:
                request = EMIRequest(
                    principal=principal,
                    annual_rate_pct=annual_rate_pct,
                    tenure_months=tenure_months,
                )
                result = calculate_emi(request)
            except Exception as exc:
                self._record_step("calculate_emi", "error", str(exc), timer.elapsed_ms)
                return self._make_result("error", error=str(exc))

        self._record_step(
            "calculate_emi",
            "success",
            f"emi={result.emi} total={result.total_payment}",
            timer.elapsed_ms,
        )

        return self._make_result(
            "completed",
            response={
                "emi": str(result.emi),
                "total_payment": str(result.total_payment),
                "total_interest": str(result.total_interest),
            },
        )

    def _call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        limit_hit = self._check_step_limit()
        if limit_hit:
            return None

        with AuditTimer() as timer:
            try:
                result = invoke_tool(tool_name, **kwargs)
            except (AuthorizationDenied, ToolPermissionDenied) as exc:
                self._record_step(tool_name, "denied", str(exc), timer.elapsed_ms)
                return None
            except CustomerNotFound as exc:
                self._record_step(tool_name, "error", str(exc), timer.elapsed_ms)
                return None
            except Exception as exc:
                self._record_step(tool_name, "error", str(exc), timer.elapsed_ms)
                return None

        self._record_step(
            tool_name, "success", f"{tool_name} returned successfully", timer.elapsed_ms,
        )
        return result
