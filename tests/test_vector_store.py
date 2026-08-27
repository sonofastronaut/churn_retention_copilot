from churn_copilot.schemas import RiskFactor, RiskProfile
from churn_copilot.vector_store import (
    retrieve_policies_semantic,
    search_policies,
)


def test_semantic_retrieval_query(monkeypatch):
    risk_profile = RiskProfile(
        churn_probability=0.0052,
        risk_drivers=[
            RiskFactor(
                feature="numdayscontractequipmentplanexpiring",
                value=30,
                shap_value=0.67,
                direction="increases_risk",
            ),
        ],
        protective_factors=[],
    )

    captured = {}

    def fake_search_policies(query: str, top_k: int):
        captured["query"] = query
        captured["top_k"] = top_k

        return [
            {"id": "equipment_plan_expiring"}
        ]

    monkeypatch.setattr(
        "churn_copilot.vector_store.search_policies",
        fake_search_policies,
    )

    policies = retrieve_policies_semantic(
        risk_profile,
        top_k=3,
    )

    assert policies[0]["id"] == "equipment_plan_expiring"
    assert captured["top_k"] == 3
    assert "0.52%" in captured["query"]
    assert "days until equipment plan expiration" in captured["query"]


def test_search_policies_accepts_relevant_query(monkeypatch):
    def fake_search_policy_matches(query: str, top_k: int):
        return [
            (
                {"id": "payment_issue"},
                0.50,
            ),
            (
                {"id": "service_quality"},
                0.18,
            ),
        ]

    monkeypatch.setattr(
        "churn_copilot.vector_store.search_policy_matches",
        fake_search_policy_matches,
    )

    policies = search_policies(
        "payment problem",
        top_k=2,
    )

    assert [policy["id"] for policy in policies] == [
        "payment_issue",
        "service_quality",
    ]


def test_search_policies_rejects_irrelevant_query(monkeypatch):
    def fake_search_policy_matches(query: str, top_k: int):
        return [
            (
                {"id": "service_quality"},
                0.14,
            ),
            (
                {"id": "low_risk_customer"},
                0.09,
            ),
        ]

    monkeypatch.setattr(
        "churn_copilot.vector_store.search_policy_matches",
        fake_search_policy_matches,
    )

    policies = search_policies(
        "change app color",
        top_k=2,
    )

    assert policies == []