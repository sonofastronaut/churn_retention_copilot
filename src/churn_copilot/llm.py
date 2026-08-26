import os
from dotenv import load_dotenv
from openai import OpenAI

from churn_copilot.schemas import RiskProfile, RetentionRecommendation, CustomerAnalysis
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_retention_prompt(
    risk_profile: RiskProfile,
    policies: list[dict],
) -> str:
    return f"""
You are a customer retention assistant.

Analyze the churn risk profile below and prepare a retention recommendation.

Risk profile:
{risk_profile.model_dump_json(indent=2)}

Allowed retention policies:
{json.dumps(policies, indent=2)}

Your task:
1. Briefly summarize the customer's churn risk.
2. Identify the main reasons behind the prediction.
3. Recommend an appropriate retention action.
4. Draft a short message to the customer.

Use only retention actions allowed by the provided policies.
Respect all policy restrictions.
Do not invent discounts, payment plans, incentives, or operational actions
that are not explicitly allowed by the retrieved policies.

Do not describe feature values as high, low, large, small, unusual,
good, or bad unless such interpretation is explicitly provided.

Treat SHAP directions only as model behavior:
- risk_drivers pushed the model prediction toward churn;
- protective_factors pushed the model prediction away from churn.

Do not claim that a feature is inherently good or bad for churn.
Do not infer causal relationships from SHAP values.
""".strip()

def generate_recommendation(
    risk_profile: RiskProfile,
    policies: list[dict],
) -> RetentionRecommendation:
    prompt = build_retention_prompt(
        risk_profile,
        policies,
    )

    response = client.responses.parse(
        model="gpt-5.6-luna",
        input=prompt,
        text_format=RetentionRecommendation,
    )

    return response.output_parsed


def answer_followup(
    question: str,
    analysis: CustomerAnalysis,
    chat_history: list[dict],
) -> str:
    instructions = f"""
You are a customer retention copilot.

The customer has already been analyzed.

Current analysis:
{analysis.model_dump_json(indent=2)}

Answer questions using the current analysis and the conversation history.

The current analysis contains retrieved_policies.

When recommending retention actions:
- use only actions explicitly allowed by retrieved_policies;
- respect all policy restrictions;
- do not invent discounts, incentives, payment plans, or operational actions
  that are not explicitly allowed;
- if the user asks for an action that violates the policies, explain that
  it is not allowed by the retrieved policies and suggest an allowed
  alternative.

You may explain the model prediction, SHAP factors, and retention recommendation.

Do not invent customer facts.
Do not infer causal relationships from SHAP values.
Do not describe feature values as high, low, good, or bad unless that
interpretation is explicitly available in the analysis.

If the user refers to something said earlier, use the conversation history
to understand the reference.
""".strip()

    messages = [
        {
            "role": "developer",
            "content": instructions,
        }
    ]

    messages.extend(chat_history)

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=messages,
    )

    return response.output_text