from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier
from functools import lru_cache
from churn_copilot.schemas import CustomerFeatures


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "churn_model.cbm"


FEATURES = [
    "age",
    "annualincome",
    "calldroprate",
    "callfailurerate",
    "monthlybilledamount",
    "numberofcomplaints",
    "numberofmonthunpaid",
    "numdayscontractequipmentplanexpiring",
    "penaltytoswitch",
    "totalminsusedinlastmonth",
    "unpaidbalance",
    "percentagecalloutsidenetwork",
    "totalcallduration",
    "avgcallduration",
]



@lru_cache(maxsize=1)
def load_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

def customer_to_dataframe(
    customer: CustomerFeatures,
) -> pd.DataFrame:
    return pd.DataFrame(
        [customer.model_dump()]
    )[FEATURES]

def predict_churn_probability(
    customer: CustomerFeatures,
) -> float:
    model = load_model()
    customer_df = customer_to_dataframe(customer)

    probability = model.predict_proba(customer_df)[0, 1]

    return float(probability)
