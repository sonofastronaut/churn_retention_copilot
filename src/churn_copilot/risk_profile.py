from churn_copilot.model import predict_churn_probability
from churn_copilot.explainer import get_factor_groups
from churn_copilot.schemas import CustomerFeatures, RiskProfile


def build_risk_profile(
    customer: CustomerFeatures,
) -> RiskProfile:
    probability = predict_churn_probability(customer)

    risk_drivers, protective_factors = get_factor_groups(customer)

    return RiskProfile(
        churn_probability=probability,
        risk_drivers=risk_drivers,
        protective_factors=protective_factors,
    )