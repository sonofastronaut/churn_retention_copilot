from churn_copilot.schemas import RiskFactor, RiskProfile
from churn_copilot.vector_store import retrieve_policies_semantic


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