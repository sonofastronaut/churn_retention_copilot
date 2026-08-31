from __future__ import annotations

import importlib
import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI

from churn_copilot.config import RETRIEVAL_MODE
from churn_copilot.schemas import (
    CustomerAnalysis,
    CustomerFeatures,
    FollowupRequest,
    FollowupResponse,
)
from churn_copilot.service import (
    analyze_customer,
)


logger = logging.getLogger(
    "uvicorn.error"
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    started = perf_counter()

    logger.info(
        "Warming up ML backend..."
    )

    # Heavy imports happen during application
    # startup instead of the first user request.
    from churn_copilot.explainer import (
        get_explainer,
    )
    from churn_copilot.model import (
        load_model,
    )

    # Import the LLM module as well so the first
    # /analyze request does not pay its import cost.
    #
    # This does NOT make an OpenAI network request.
    importlib.import_module(
        "churn_copilot.llm"
    )

    step_started = perf_counter()

    load_model()

    logger.info(
        "model_warmup_duration=%.3fs",
        perf_counter() - step_started,
    )

    step_started = perf_counter()

    get_explainer()

    logger.info(
        "shap_warmup_duration=%.3fs",
        perf_counter() - step_started,
    )

    logger.info(
        "Backend warmup complete in %.3fs",
        perf_counter() - started,
    )

    logger.info(
        "retrieval_mode=%s",
        RETRIEVAL_MODE,
    )

    yield


app = FastAPI(
    title="Churn Retention Copilot API",
    version="0.2.0",
    lifespan=lifespan,
)


def answer_followup(
    question: str,
    analysis: CustomerAnalysis,
    chat_history: list[dict],
) -> str:
    """
    Lazy proxy for the real LLM function.

    Kept at module level so tests can monkeypatch:
        churn_copilot.api.answer_followup
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