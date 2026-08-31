from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from churn_copilot.schemas import (
    RiskProfile,
)


POLICIES_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "retention_policies.json"
)


@lru_cache(maxsize=1)
def load_policies() -> list[dict]:
    with open(
        POLICIES_PATH,
        encoding="utf-8",
    ) as file:
        return json.load(file)


def retrieve_policies(
    risk_profile: RiskProfile,
) -> list[dict]:
    policies = load_policies()

    selected_ids = {
        "low_risk_customer",
    }

    risk_features = {
        factor.feature
        for factor in risk_profile.risk_drivers
    }

    if (
        "numdayscontractequipmentplanexpiring"
        in risk_features
    ):
        selected_ids.add(
            "equipment_plan_expiring"
        )

    if (
        "unpaidbalance" in risk_features
        or "numberofmonthunpaid"
        in risk_features
    ):
        selected_ids.add(
            "payment_issue"
        )

    if (
        "callfailurerate" in risk_features
        or "calldroprate" in risk_features
    ):
        selected_ids.add(
            "service_quality"
        )

    return [
        policy
        for policy in policies
        if policy["id"] in selected_ids
    ]