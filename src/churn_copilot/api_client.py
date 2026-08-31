from __future__ import annotations

from functools import lru_cache

import httpx

from churn_copilot.config import (
    API_CONNECT_TIMEOUT,
    API_POOL_TIMEOUT,
    API_READ_TIMEOUT,
    API_URL,
    API_WRITE_TIMEOUT,
)
from churn_copilot.schemas import (
    CustomerAnalysis,
    CustomerFeatures,
    FollowupResponse,
)


@lru_cache(maxsize=1)
def get_api_client() -> httpx.Client:
    return httpx.Client(
        base_url=API_URL,
        timeout=httpx.Timeout(
            connect=API_CONNECT_TIMEOUT,
            read=API_READ_TIMEOUT,
            write=API_WRITE_TIMEOUT,
            pool=API_POOL_TIMEOUT,
        ),
    )


def health_check() -> bool:
    try:
        response = get_api_client().get(
            "/health"
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return False

    return response.json().get("status") == "ok"


def analyze_customer(
    customer: CustomerFeatures,
) -> CustomerAnalysis:
    response = get_api_client().post(
        "/analyze",
        json=customer.model_dump(),
    )

    response.raise_for_status()

    return CustomerAnalysis.model_validate(
        response.json()
    )


def answer_followup(
    question: str,
    analysis: CustomerAnalysis,
    chat_history: list[dict],
) -> str:
    response = get_api_client().post(
        "/chat",
        json={
            "question": question,
            "analysis": analysis.model_dump(),
            "chat_history": chat_history,
        },
    )

    response.raise_for_status()

    result = FollowupResponse.model_validate(
        response.json()
    )

    return result.answer