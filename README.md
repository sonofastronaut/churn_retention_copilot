# Churn Retention Copilot

[English](README.md) | [Русский](README_RU.md)

An end-to-end AI copilot for customer churn analysis and retention decision support.

The project combines machine learning, explainability, semantic RAG, structured LLM outputs, multi-turn chat, FastAPI, Streamlit, and Docker in a single application.

## What the Project Does

Given a customer profile, the system:

1. Predicts churn probability using CatBoost.
2. Explains the prediction with SHAP.
3. Builds a structured customer risk profile.
4. Retrieves relevant retention policies using semantic search.
5. Generates a policy-grounded retention recommendation with an LLM.
6. Supports follow-up questions in a multi-turn conversation.
7. Exposes the pipeline through both Streamlit and FastAPI.

## Architecture

```text
Customer Features
       |
       v
    CatBoost
       |
       v
Churn Probability
       |
       v
      SHAP
       |
       v
   Risk Profile
       |
       v
SentenceTransformer
       |
       v
     Qdrant
       |
       v
Relevant Retention Policies
       |
       v
      LLM
       |
       v
Structured Recommendation
       |
       +------> Streamlit UI
       |
       +------> FastAPI
```

## Tech Stack

- Python 3.12
- CatBoost
- SHAP
- Pydantic
- OpenAI API
- Sentence Transformers
- Qdrant
- FastAPI
- Streamlit
- Docker

## Machine Learning and Explainability

CatBoost is used to estimate the probability that a customer will churn.

After prediction, SHAP is used to explain how individual features influenced the model output.

The factors are separated into two groups:

- `risk_drivers` — features that pushed the model prediction toward churn;
- `protective_factors` — features that pushed the prediction away from churn.

SHAP values are treated as explanations of model behavior rather than evidence of causal relationships.

For example:

```text
Risk drivers:
- days until equipment plan expiration
- average call duration
- total call duration
- annual income
- call failure rate

Protective factors:
- age
- unpaid balance
- penalty to switch
- number of months unpaid
- number of complaints
```

The exact factors depend on the customer profile and model prediction.

## Semantic RAG

Retention policies are stored as structured documents.

Each policy contains:

- a description of when it applies;
- allowed retention actions;
- operational restrictions.

The policy documents are converted into embeddings using:

```text
all-MiniLM-L6-v2
```

The embeddings are stored in a local Qdrant vector database.

For each customer, the application converts the `RiskProfile` into a natural-language retrieval query containing the predicted churn probability and the main risk drivers.

Example:

```text
Customer retention situation.
Predicted churn probability: 0.52%.
Factors pushing the churn prediction upward:
days until equipment plan expiration with value 30;
average call duration with value 700;
...
```

Qdrant performs semantic similarity search and returns the most relevant retention policies.

These retrieved policies are then provided to the LLM as operational context.

## Policy-Grounded LLM Recommendations

The LLM is instructed to:

- use only retention actions explicitly allowed by retrieved policies;
- respect all policy restrictions;
- avoid inventing discounts, payment plans, incentives, or operational actions;
- avoid treating SHAP associations as causal relationships;
- avoid interpreting feature values as inherently good or bad without supporting context.

For example, if the user asks:

```text
Can we offer a 50% discount?
```

and no retrieved policy authorizes a discount, the copilot rejects the action and suggests an allowed alternative instead.

Example response:

```text
No. A 50% discount is not allowed under the retrieved policies.

Instead, the customer can receive a light-touch renewal reminder,
assistance reviewing renewal options, an account and plan review,
or a service quality review if relevant.
```

## Structured LLM Output

The initial recommendation is returned as a validated Pydantic object.

Example structure:

```json
{
  "summary": "Brief churn risk summary",
  "main_reasons": [
    "Reason 1",
    "Reason 2"
  ],
  "recommended_action": "Recommended retention action",
  "customer_message": "Suggested message to the customer"
}
```

This makes the LLM output easier to validate, display in the UI, and expose through the API.

## Multi-Turn Copilot

After the initial customer analysis, users can ask follow-up questions such as:

```text
Why is the churn risk so low?
```

```text
Which factor matters the most?
```

```text
Can you explain that in more detail?
```

```text
Can we offer this customer a discount?
```

The assistant receives:

- the current customer analysis;
- retrieved retention policies;
- previous conversation history.

This allows it to maintain conversational context while remaining grounded in the current customer analysis and policy constraints.

## FastAPI

The project exposes the main functionality as a REST API.

Start the API:

```bash
python -m uvicorn churn_copilot.api:app --reload
```

Swagger UI is available at:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

```text
GET  /health
POST /analyze
POST /chat
```

### `GET /health`

Simple API health check.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /analyze`

Accepts customer features and runs the full analysis pipeline:

```text
CustomerFeatures
       ↓
CatBoost
       ↓
SHAP
       ↓
RiskProfile
       ↓
Semantic RAG
       ↓
LLM
       ↓
CustomerAnalysis
```

The response contains:

- churn probability;
- risk drivers;
- protective factors;
- retrieved retention policies;
- structured retention recommendation;
- suggested customer message.

### `POST /chat`

Handles follow-up questions about an existing customer analysis.

The request contains:

- the user question;
- the current `CustomerAnalysis`;
- previous chat history.

This endpoint enables the multi-turn copilot behavior outside the Streamlit application.

## Streamlit Application

Start the UI with:

```bash
streamlit run app.py
```

The interface allows users to:

- configure customer features;
- run churn analysis;
- view churn probability;
- review the main reasons behind the prediction;
- receive a retention recommendation;
- view a suggested customer message;
- inspect technical SHAP details;
- ask follow-up questions in a chat interface.

## Docker

Build the Docker image:

```bash
docker build -t churn-retention-copilot .
```

Run the API container:

```bash
docker run --rm --env-file .env -p 8000:8000 churn-retention-copilot
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The Qdrant policy index is created during the Docker image build process.

The OpenAI API key is supplied at container runtime through the `.env` file and is not embedded into the Docker image.

## Installation

Clone the repository:

```bash
git clone https://github.com/sonofastronaut/churn_retentoin_copilot.git
cd churn_retentoin_copilot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the project in editable mode:

```bash
python -m pip install -e .
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```text
OPENAI_API_KEY=your_api_key_here
```

The real `.env` file is excluded from Git and Docker build context.

## Building the Vector Index

Retention policies are stored in:

```text
data/retention_policies.json
```

To build the local Qdrant vector index manually:

```bash
python -m churn_copilot.vector_store
```

The generated local Qdrant data is excluded from Git.

## Project Structure

```text
churn_retention_copilot/
├── data/
│   └── retention_policies.json
├── models/
│   └── churn_model.cbm
├── src/
│   └── churn_copilot/
│       ├── __init__.py
│       ├── api.py
│       ├── explainer.py
│       ├── llm.py
│       ├── model.py
│       ├── retriever.py
│       ├── risk_profile.py
│       ├── schemas.py
│       ├── service.py
│       └── vector_store.py
├── app.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── test_model.py
```

## End-to-End Pipeline

```text
Customer
   ↓
Pydantic Validation
   ↓
CatBoost Prediction
   ↓
SHAP Explanation
   ↓
RiskProfile
   ↓
Semantic Embedding
   ↓
Qdrant Retrieval
   ↓
Retention Policies
   ↓
LLM Recommendation
   ↓
Structured Pydantic Output
   ↓
Streamlit / FastAPI / Multi-turn Chat
```

## Key Engineering Features

The project demonstrates:

- end-to-end ML inference;
- explainable ML with SHAP;
- semantic retrieval with vector embeddings;
- vector search with Qdrant;
- retrieval-augmented generation;
- grounded LLM recommendations;
- structured LLM output with Pydantic;
- multi-turn conversational context;
- REST API design with FastAPI;
- interactive UI with Streamlit;
- Docker containerization;
- Python package structure with `pyproject.toml`.

## Disclaimer

The retention policies included in this repository are synthetic demonstration examples created specifically for this project.

They do not represent real policies, rules, or procedures of any telecom operator or other company.