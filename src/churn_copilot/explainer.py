import pandas as pd
import shap
from functools import lru_cache
from churn_copilot.schemas import CustomerFeatures

from churn_copilot.model import (
    FEATURES,
    customer_to_dataframe,
    load_model,
)


@lru_cache(maxsize=1)
def get_explainer():
    model=load_model()
    return shap.TreeExplainer(model)

def get_shap_values(customer: CustomerFeatures) -> dict:
    customer_df = customer_to_dataframe(customer)

    explainer = get_explainer()
    shap_values = explainer.shap_values(customer_df)

    return {
        feature: float(value)
        for feature, value in zip(FEATURES, shap_values[0])
    }


def get_factor_groups(
    customer: CustomerFeatures,
    top_n: int = 5,
) -> tuple[list[dict], list[dict]]:
    shap_values = get_shap_values(customer)

    risk_drivers = []
    protective_factors = []

    for feature, shap_value in shap_values.items():
        factor = {
            "feature": feature,
            "value": getattr(customer, feature),
            "shap_value": shap_value,
            "direction": (
                "increases_risk"
                if shap_value > 0
                else "decreases_risk"
            ),
        }

        if shap_value > 0:
            risk_drivers.append(factor)
        else:
            protective_factors.append(factor)

    risk_drivers.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    protective_factors.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return (
        risk_drivers[:top_n],
        protective_factors[:top_n],
    )