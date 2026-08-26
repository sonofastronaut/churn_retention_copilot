from churn_copilot.risk_profile import build_risk_profile
from churn_copilot.schemas import CustomerFeatures


def test_build_risk_profile():
    customer = CustomerFeatures(
        age=45,
        annualincome=120000,
        calldroprate=0.03,
        callfailurerate=0.01,
        monthlybilledamount=70,
        numberofcomplaints=2,
        numberofmonthunpaid=1,
        numdayscontractequipmentplanexpiring=30,
        penaltytoswitch=200,
        totalminsusedinlastmonth=250,
        unpaidbalance=100,
        percentagecalloutsidenetwork=0.4,
        totalcallduration=3500,
        avgcallduration=700,
    )

    profile = build_risk_profile(customer)

    assert 0.0 <= profile.churn_probability <= 1.0

    assert len(profile.risk_drivers) > 0
    assert len(profile.protective_factors) > 0

    assert all(
        factor.shap_value > 0
        for factor in profile.risk_drivers
    )

    assert all(
        factor.shap_value < 0
        for factor in profile.protective_factors
    )