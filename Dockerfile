FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml ./

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY data/retention_policies.json ./data/retention_policies.json

RUN python -m pip install -e .

RUN python -c "from churn_copilot.vector_store import build_policy_index; build_policy_index()"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "churn_copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]