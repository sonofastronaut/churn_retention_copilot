from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


API_URL = os.getenv(
    "CHURN_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


RETRIEVAL_MODE = os.getenv(
    "RETRIEVAL_MODE",
    "rules",
).strip().lower()


VALID_RETRIEVAL_MODES = {
    "rules",
    "semantic",
}


if RETRIEVAL_MODE not in VALID_RETRIEVAL_MODES:
    raise ValueError(
        "RETRIEVAL_MODE must be one of: "
        f"{', '.join(sorted(VALID_RETRIEVAL_MODES))}. "
        f"Received: {RETRIEVAL_MODE!r}"
    )


API_CONNECT_TIMEOUT = float(
    os.getenv(
        "API_CONNECT_TIMEOUT",
        "3",
    )
)

API_READ_TIMEOUT = float(
    os.getenv(
        "API_READ_TIMEOUT",
        "90",
    )
)

API_WRITE_TIMEOUT = float(
    os.getenv(
        "API_WRITE_TIMEOUT",
        "10",
    )
)

API_POOL_TIMEOUT = float(
    os.getenv(
        "API_POOL_TIMEOUT",
        "5",
    )
)