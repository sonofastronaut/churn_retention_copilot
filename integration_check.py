import sys

sys.path.insert(0, "src")

from churn_copilot.model import predict_churn_probability
from churn_copilot.explainer import get_shap_values, get_factor_groups
from churn_copilot.risk_profile import build_risk_profile
from churn_copilot.llm import build_retention_prompt, generate_recommendation
from churn_copilot.schemas import CustomerFeatures
from churn_copilot.vector_store import retrieve_policies_semantic



customer = {
    "age": 45,
    "annualincome": 120000,
    "calldroprate": 0.03,
    "callfailurerate": 0.01,
    "monthlybilledamount": 70,
    "numberofcomplaints": 2,
    "numberofmonthunpaid": 1,
    "numdayscontractequipmentplanexpiring": 30,
    "penaltytoswitch": 200,
    "totalminsusedinlastmonth": 250,
    "unpaidbalance": 100,
    "percentagecalloutsidenetwork": 0.4,
    "totalcallduration": 3500,
    "avgcallduration": 700,
}
customer = CustomerFeatures(**customer)

probability = predict_churn_probability(customer)

print(probability)

shap_values = get_shap_values(customer)

for feature, value in shap_values.items():
    print(f"{feature}: {value:.4f}")

risk_drivers, protective_factors = get_factor_groups(customer)

print("\nRisk drivers:")
for factor in risk_drivers:
    print(
        factor["feature"],
        factor["value"],
        round(factor["shap_value"], 4),
    )

print("\nProtective factors:")
for factor in protective_factors:
    print(
        factor["feature"],
        factor["value"],
        round(factor["shap_value"], 4),
    )

risk_profile = build_risk_profile(customer)
print("\nRisk profile:")
print(risk_profile)

policies = retrieve_policies_semantic(risk_profile)

print("\nRetrieved policies:")
for policy in policies:
    print(policy["id"])

print("\nRisk Profile JSON")
print(risk_profile.model_dump_json(indent=2))




prompt = build_retention_prompt(
    risk_profile,
    policies,
)
print("\nRetention Prompt:")
print(prompt)

recommendation = generate_recommendation(
    risk_profile,
    policies,
)

print("\nStructured recommendation:")
print(recommendation)

print("\nAs JSON:")
print(recommendation.model_dump_json(indent=2))

