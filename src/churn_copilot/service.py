from __future__ import annotations

import logging
from time import perf_counter

from churn_copilot.config import RETRIEVAL_MODE
from churn_copilot.schemas import (
    CustomerAnalysis,
    CustomerFeatures,
    RiskProfile,
)


logger = logging.getLogger(
    "uvicorn.error"
)


def retrieve_policies(
    risk_profile: RiskProfile,
) -> list[dict]:
    if RETRIEVAL_MODE == "rules":
        from churn_copilot.retriever import (
            retrieve_policies as retrieve_policies_rules,
        )

        return retrieve_policies_rules(
            risk_profile
        )

    if RETRIEVAL_MODE == "semantic":
        from churn_copilot.vector_store import (
            retrieve_policies_semantic,
        )

        return retrieve_policies_semantic(
            risk_profile
        )

    raise RuntimeError(
        "Unsupported retrieval mode: "
        f"{RETRIEVAL_MODE!r}"
    )


def analyze_customer(
    customer: CustomerFeatures,
) -> CustomerAnalysis:
    started = perf_counter()

    import_started = perf_counter()

    from churn_copilot.llm import (
        generate_recommendation,
    )
    from churn_copilot.risk_profile import (
        build_risk_profile,
    )

    logger.info(
        "request_imports_duration=%.3fs",
        perf_counter() - import_started,
    )

    step_started = perf_counter()

    risk_profile = build_risk_profile(
        customer
    )

    logger.info(
        "risk_profile_duration=%.3fs",
        perf_counter() - step_started,
    )

    step_started = perf_counter()

    policies = retrieve_policies(
        risk_profile
    )

    logger.info(
        "retrieval_duration=%.3fs mode=%s",
        perf_counter() - step_started,
        RETRIEVAL_MODE,
    )

    step_started = perf_counter()

    recommendation = generate_recommendation(
        risk_profile,
        policies,
    )

    logger.info(
        "llm_duration=%.3fs",
        perf_counter() - step_started,
    )

    result = CustomerAnalysis(
        risk_profile=risk_profile,
        recommendation=recommendation,
        retrieved_policies=policies,
    )

    logger.info(
        "analyze_customer_duration=%.3fs",
        perf_counter() - started,
    )

    return result