from pydantic import BaseModel, ConfigDict, Field

class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    annualincome: float
    calldroprate: float
    callfailurerate: float
    monthlybilledamount: float
    numberofcomplaints: float
    numberofmonthunpaid: float
    numdayscontractequipmentplanexpiring: float
    penaltytoswitch: float
    totalminsusedinlastmonth: float
    unpaidbalance: float
    percentagecalloutsidenetwork: float
    totalcallduration: float
    avgcallduration: float

class RiskFactor(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str

class RiskProfile(BaseModel):
    churn_probability: float
    risk_drivers: list[RiskFactor]
    protective_factors: list[RiskFactor]


class RetentionRecommendation(BaseModel):
    summary: str
    main_reasons: list[str]
    recommended_action: str
    customer_message: str


class CustomerAnalysis(BaseModel):
    risk_profile: RiskProfile
    recommendation: RetentionRecommendation
    retrieved_policies: list[dict]


class ChatMessage(BaseModel):
    role: str
    content: str


class FollowupRequest(BaseModel):
    question: str
    analysis: CustomerAnalysis
    chat_history: list[ChatMessage] = Field(
        default_factory=list
    )

class FollowupResponse(BaseModel):
    answer: str