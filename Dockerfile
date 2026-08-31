FROM python:3.12-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


COPY requirements.txt pyproject.toml ./


RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --no-cache-dir \
        -r requirements.txt


COPY src ./src
COPY models ./models
COPY data/retention_policies.json \
    ./data/retention_policies.json


RUN python -m pip install -e .


# By default we do NOT build the semantic index.
#
# To build an image with semantic retrieval:
#
# docker build \
#   --build-arg BUILD_SEMANTIC_INDEX=1 \
#   -t churn-copilot .
#
ARG BUILD_SEMANTIC_INDEX=0


RUN if [ "$BUILD_SEMANTIC_INDEX" = "1" ]; then \
        python -m churn_copilot.vector_store; \
    fi


EXPOSE 8000


CMD [
    "python",
    "-m",
    "uvicorn",
    "churn_copilot.api:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]