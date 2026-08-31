from fastapi import FastAPI

from churn_copilot.schemas import (
    CustomerAnalysis,
    CustomerFeatures,
    FollowupRequest,
    FollowupResponse,
)
from churn_copilot.service import (
    analyze_customer,
)


app = FastAPI(
    title="Churn Retention Copilot API",
    version="0.2.0",
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
    # Lazy import.
    #
    # API startup should not load LLM-related
    # code until /chat is actually called.
    from churn_copilot.llm import (
        answer_followup,
    )

    chat_history = [
        message.model_dump()
        for message
        in request.chat_history
    ]

    answer = answer_followup(
        question=request.question,
        analysis=request.analysis,
        chat_history=chat_history,
    )

    return FollowupResponse(
        answer=answer,
    )
