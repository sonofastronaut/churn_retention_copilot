from churn_copilot.model import predict_churn_probability
from churn_copilot.schemas import CustomerFeatures


def test_predict_churn_probability():
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

    probability = predict_churn_probability(customer)

    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0