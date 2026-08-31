from __future__ import annotations

from fastapi import FastAPI

from churn_copilot.schemas import (
    CustomerAnalysis,
    CustomerFeatures,
    FollowupRequest,
    FollowupResponse,
)
from churn_copilot.service import analyze_customer


app = FastAPI(
    title="Churn Retention Copilot API",
    version="0.2.0",
)


def answer_followup(
    question: str,
    analysis: CustomerAnalysis,
    chat_history: list[dict],
) -> str:
    """
    Lazy proxy around the real LLM function.

    Keeping this function at module level has two benefits:

    1. Importing churn_copilot.api does not eagerly import
       the OpenAI/LLM stack.
    2. Tests can still monkeypatch
       churn_copilot.api.answer_followup.
    """
    from churn_copilot.llm import (
        answer_followup as llm_answer_followup,
    )

    return llm_answer_followup(
        question=question,
        analysis=analysis,
        chat_history=chat_history,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post(
    "/analyze",
    response_model=CustomerAnalysis,
)
def analyze(
    customer: CustomerFeatures,
) -> CustomerAnalysis:
    return analyze_customer(
        customer
    )


@app.post(
    "/chat",
    response_model=FollowupResponse,
)
def chat(
    request: FollowupRequest,
) -> FollowupResponse:
    chat_history = [
        message.model_dump()
        for message in request.chat_history
    ]

    answer = answer_followup(
        question=request.question,
        analysis=request.analysis,
        chat_history=chat_history,
    )

    return FollowupResponse(
        answer=answer,
    )