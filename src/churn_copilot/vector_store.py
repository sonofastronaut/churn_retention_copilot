from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from churn_copilot.retriever import (
    load_policies,
)
from churn_copilot.schemas import (
    RiskProfile,
)


if TYPE_CHECKING:
    from qdrant_client import (
        QdrantClient,
    )
    from sentence_transformers import (
        SentenceTransformer,
    )


FEATURE_LABELS = {
    "numdayscontractequipmentplanexpiring": (
        "days until equipment plan expiration"
    ),
    "avgcallduration": (
        "average call duration"
    ),
    "totalcallduration": (
        "total call duration"
    ),
    "annualincome": (
        "annual income"
    ),
    "callfailurerate": (
        "call failure rate"
    ),
    "calldroprate": (
        "call drop rate"
    ),
    "unpaidbalance": (
        "unpaid balance"
    ),
    "numberofmonthunpaid": (
        "months unpaid"
    ),
    "numberofcomplaints": (
        "number of complaints"
    ),
}


QDRANT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "qdrant"
)


COLLECTION_NAME = (
    "retention_policies"
)


EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)


MIN_RELEVANCE_SCORE = 0.30


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Lazy import is intentional.
    #
    # In rules mode torch / transformers
    # should never be imported.
    from sentence_transformers import (
        SentenceTransformer,
    )

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    # Qdrant is also imported only when
    # semantic retrieval is actually used.
    from qdrant_client import (
        QdrantClient,
    )

    return QdrantClient(
        path=str(QDRANT_PATH)
    )


def policy_to_text(
    policy: dict,
) -> str:
    allowed_actions = ", ".join(
        policy["allowed_actions"]
    )

    restrictions = ", ".join(
        policy["restrictions"]
    )

    return (
        f"Title: {policy['title']}. "
        f"Description: "
        f"{policy['description']} "
        f"Allowed actions: "
        f"{allowed_actions}. "
        f"Restrictions: "
        f"{restrictions}."
    )


def build_policy_index() -> None:
    # Imported only when the index
    # is explicitly being built.
    from qdrant_client import models

    policies = load_policies()

    embedding_model = (
        get_embedding_model()
    )

    texts = [
        policy_to_text(policy)
        for policy in policies
    ]

    embeddings = (
        embedding_model.encode(
            texts
        )
    )

    client = get_qdrant_client()

    if client.collection_exists(
        COLLECTION_NAME
    ):
        client.delete_collection(
            COLLECTION_NAME
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embeddings.shape[1],
            distance=(
                models.Distance.COSINE
            ),
        ),
    )

    points = [
        models.PointStruct(
            id=index,
            vector=embedding.tolist(),
            payload=policy,
        )
        for index, (
            policy,
            embedding,
        )
        in enumerate(
            zip(
                policies,
                embeddings,
            )
        )
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


def search_policy_matches(
    query: str,
    top_k: int = 3,
) -> list[tuple[dict, float]]:
    embedding_model = (
        get_embedding_model()
    )

    query_vector = (
        embedding_model.encode(
            query
        ).tolist()
    )

    client = get_qdrant_client()

    if not client.collection_exists(
        COLLECTION_NAME
    ):
        raise RuntimeError(
            "Qdrant policy index does not exist. "
            "Build it with: "
            "python -m "
            "churn_copilot.vector_store"
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        (
            result.payload,
            float(result.score),
        )
        for result in results
        if result.payload is not None
    ]


def search_policies(
    query: str,
    top_k: int = 3,
) -> list[dict]:
    matches = search_policy_matches(
        query=query,
        top_k=top_k,
    )

    if not matches:
        return []

    top_score = matches[0][1]

    if (
        top_score
        < MIN_RELEVANCE_SCORE
    ):
        return []

    return [
        policy
        for policy, _ in matches
    ]


def retrieve_policies_semantic(
    risk_profile: RiskProfile,
    top_k: int = 3,
) -> list[dict]:
    risk_drivers = []

    for factor in risk_profile.risk_drivers:
        label = FEATURE_LABELS.get(
            factor.feature,
            factor.feature,
        )

        risk_drivers.append(
            f"{label} with value {factor.value}"
        )

    query = (
        "Customer retention situation. "
        "Predicted churn probability: "
        f"{risk_profile.churn_probability:.2%}. "
        "Factors pushing the churn prediction "
        "upward: "
        + "; ".join(risk_drivers)
    )

    return search_policies(
        query=query,
        top_k=top_k,
    )