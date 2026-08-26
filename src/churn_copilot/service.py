from churn_copilot.llm import generate_recommendation
from churn_copilot.risk_profile import build_risk_profile
from churn_copilot.schemas import (
    CustomerAnalysis,
    CustomerFeatures,
)
from churn_copilot.vector_store import retrieve_policies_semantic


def analyze_customer(
    customer: CustomerFeatures,
) -> CustomerAnalysis:
    risk_profile = build_risk_profile(customer)

    policies = retrieve_policies_semantic(risk_profile)

    recommendation = generate_recommendation(
        risk_profile,
        policies,
    )

    return CustomerAnalysis(
    risk_profile=risk_profile,
    recommendation=recommendation,
    retrieved_policies=policies,
)