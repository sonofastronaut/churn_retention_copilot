from fastapi.testclient import TestClient

from churn_copilot.api import app
from churn_copilot.schemas import (
    CustomerAnalysis,
    RetentionRecommendation,
    RiskFactor,
    RiskProfile,
)


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint(monkeypatch):
    fake_analysis = CustomerAnalysis(
        risk_profile=RiskProfile(
            churn_probability=0.25,
            risk_drivers=[
                RiskFactor(
                    feature="callfailurerate",
                    value=0.1,
                    shap_value=0.5,
                    direction="increases_risk",
                )
            ],
            protective_factors=[],
        ),
        recommendation=RetentionRecommendation(
            summary="Test summary",
            main_reasons=["Test reason"],
            recommended_action="Test action",
            customer_message="Test message",
        ),
        retrieved_policies=[
            {
                "id": "service_quality",
                "title": "Service quality review",
            }
        ],
    )

    def fake_analyze_customer(customer):
        return fake_analysis

    monkeypatch.setattr(
        "churn_copilot.api.analyze_customer",
        fake_analyze_customer,
    )

    response = client.post(
        "/analyze",
        json={
            "age": 45,
            "annualincome": 120000,
            "calldroprate": 0.03,
            "callfailurerate": 0.01,
            "monthlybilledamount": 70,
            "numberofcomplaints": 2,
            "numberofmonthunpaid": 1,
            "numdayscontractequipmentplanexpiring": 30,
            "penaltytoswitch": 200,
            "totalminsusedinlastmonth": 250,
            "unpaidbalance": 100,
            "percentagecalloutsidenetwork": 0.4,
            "totalcallduration": 3500,
            "avgcallduration": 700,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["risk_profile"]["churn_probability"] == 0.25
    assert data["recommendation"]["summary"] == "Test summary"
    assert data["retrieved_policies"][0]["id"] == "service_quality"


def test_chat_endpoint(monkeypatch):
    captured = {}

    def fake_answer_followup(
        question,
        analysis,
        chat_history,
    ):
        captured["question"] = question
        captured["analysis"] = analysis
        captured["chat_history"] = chat_history

        return "Test follow-up answer"

    monkeypatch.setattr(
        "churn_copilot.api.answer_followup",
        fake_answer_followup,
    )

    response = client.post(
        "/chat",
        json={
            "question": "Can we offer a discount?",
            "analysis": {
                "risk_profile": {
                    "churn_probability": 0.25,
                    "risk_drivers": [
                        {
                            "feature": "callfailurerate",
                            "value": 0.1,
                            "shap_value": 0.5,
                            "direction": "increases_risk",
                        }
                    ],
                    "protective_factors": [],
                },
                "recommendation": {
                    "summary": "Test summary",
                    "main_reasons": ["Test reason"],
                    "recommended_action": "Test action",
                    "customer_message": "Test message",
                },
                "retrieved_policies": [
                    {
                        "id": "service_quality",
                        "title": "Service quality review",
                    }
                ],
            },
            "chat_history": [
                {
                    "role": "user",
                    "content": "What can we offer?",
                },
                {
                    "role": "assistant",
                    "content": "A service quality review.",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Test follow-up answer"
    }

    assert captured["question"] == "Can we offer a discount?"
    assert len(captured["chat_history"]) == 2
    assert captured["chat_history"][0]["role"] == "user"