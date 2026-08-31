import httpx
import streamlit as st

from churn_copilot.api_client import (
    analyze_customer,
    answer_followup,
)
from churn_copilot.config import (
    API_URL,
)
from churn_copilot.schemas import (
    CustomerFeatures,
)


st.set_page_config(
    page_title="Churn Retention Copilot",
    page_icon="🤖",
)


st.title(
    "Churn Retention Copilot"
)

st.write(
    "AI assistant for customer churn analysis "
    "and retention recommendations."
)


if (
    "chat_history"
    not in st.session_state
):
    st.session_state.chat_history = []


if (
    "result"
    not in st.session_state
):
    st.chat_message(
        "assistant"
    ).write(
        "Привет! Выбери параметры клиента, "
        "и я проанализирую риск оттока."
    )


with st.sidebar:
    st.header(
        "Customer features"
    )

    age = st.number_input(
        "Age",
        value=45.0,
    )

    annualincome = st.number_input(
        "Annual income",
        value=120000.0,
    )

    calldroprate = st.number_input(
        "Call drop rate",
        value=0.03,
        format="%.4f",
    )

    callfailurerate = st.number_input(
        "Call failure rate",
        value=0.01,
        format="%.4f",
    )

    monthlybilledamount = st.number_input(
        "Monthly billed amount",
        value=70.0,
    )

    numberofcomplaints = st.number_input(
        "Number of complaints",
        value=2.0,
    )

    numberofmonthunpaid = st.number_input(
        "Number of months unpaid",
        value=1.0,
    )

    numdayscontractequipmentplanexpiring = (
        st.number_input(
            "Days until equipment plan expires",
            value=30.0,
        )
    )

    penaltytoswitch = st.number_input(
        "Penalty to switch",
        value=200.0,
    )

    totalminsusedinlastmonth = st.number_input(
        "Minutes used last month",
        value=250.0,
    )

    unpaidbalance = st.number_input(
        "Unpaid balance",
        value=100.0,
    )

    percentagecalloutsidenetwork = (
        st.number_input(
            "Calls outside network",
            value=0.4,
            format="%.4f",
        )
    )

    totalcallduration = st.number_input(
        "Total call duration",
        value=3500.0,
    )

    avgcallduration = st.number_input(
        "Average call duration",
        value=700.0,
    )

    analyze_button = st.button(
        "Analyze customer",
        type="primary",
    )


if analyze_button:
    st.session_state.chat_history = []

    st.session_state.pop(
        "result",
        None,
    )

    customer = CustomerFeatures(
        age=age,
        annualincome=annualincome,
        calldroprate=calldroprate,
        callfailurerate=callfailurerate,
        monthlybilledamount=(
            monthlybilledamount
        ),
        numberofcomplaints=(
            numberofcomplaints
        ),
        numberofmonthunpaid=(
            numberofmonthunpaid
        ),
        numdayscontractequipmentplanexpiring=(
            numdayscontractequipmentplanexpiring
        ),
        penaltytoswitch=penaltytoswitch,
        totalminsusedinlastmonth=(
            totalminsusedinlastmonth
        ),
        unpaidbalance=unpaidbalance,
        percentagecalloutsidenetwork=(
            percentagecalloutsidenetwork
        ),
        totalcallduration=(
            totalcallduration
        ),
        avgcallduration=avgcallduration,
    )

    with st.spinner(
        "Analyzing customer..."
    ):
        try:
            result = analyze_customer(
                customer
            )

        except httpx.ConnectError:
            st.error(
                "Backend API is unavailable. "
                f"Expected API at {API_URL}"
            )

        except httpx.TimeoutException:
            st.error(
                "Backend request timed out. "
                "Check API timing logs."
            )

        except httpx.HTTPStatusError as exc:
            st.error(
                "Backend returned an error: "
                f"{exc.response.status_code}"
            )

        except httpx.HTTPError as exc:
            st.error(
                "Backend request failed: "
                f"{exc}"
            )

        else:
            st.session_state.result = (
                result
            )


if "result" in st.session_state:
    result = (
        st.session_state.result
    )

    risk = result.risk_profile

    recommendation = (
        result.recommendation
    )

    st.success(
        "Analysis complete"
    )

    with st.chat_message(
        "assistant"
    ):
        st.subheader(
            "Churn analysis"
        )

        st.metric(
            "Churn probability",
            f"{risk.churn_probability:.2%}",
        )

        st.write(
            recommendation.summary
        )

        st.markdown(
            "**Main reasons:**"
        )

        for reason in (
            recommendation.main_reasons
        ):
            st.markdown(
                f"- {reason}"
            )

        st.markdown(
            "**Recommended action:**"
        )

        st.write(
            recommendation.recommended_action
        )

        st.markdown(
            "**Suggested customer message:**"
        )

        st.info(
            recommendation.customer_message
        )

        with st.expander(
            "Model explanation"
        ):
            st.markdown(
                "**Risk drivers**"
            )

            for factor in (
                risk.risk_drivers
            ):
                st.write(
                    f"{factor.feature}: "
                    f"value={factor.value}, "
                    f"SHAP="
                    f"{factor.shap_value:.3f}"
                )

            st.markdown(
                "**Protective factors**"
            )

            for factor in (
                risk.protective_factors
            ):
                st.write(
                    f"{factor.feature}: "
                    f"value={factor.value}, "
                    f"SHAP="
                    f"{factor.shap_value:.3f}"
                )

    for message in (
        st.session_state.chat_history
    ):
        with st.chat_message(
            message["role"]
        ):
            st.write(
                message["content"]
            )

    user_question = st.chat_input(
        "Ask a follow-up question "
        "about this customer..."
    )

    if user_question:
        with st.chat_message(
            "user"
        ):
            st.write(
                user_question
            )

        with st.spinner(
            "Thinking..."
        ):
            try:
                followup_answer = (
                    answer_followup(
                        question=user_question,
                        analysis=result,
                        chat_history=(
                            st.session_state
                            .chat_history
                        ),
                    )
                )

            except httpx.ConnectError:
                st.error(
                    "Backend API is unavailable. "
                    f"Expected API at {API_URL}"
                )
                st.stop()

            except httpx.TimeoutException:
                st.error(
                    "Chat request timed out."
                )
                st.stop()

            except httpx.HTTPStatusError as exc:
                st.error(
                    "Backend returned an error: "
                    f"{exc.response.status_code}"
                )
                st.stop()

            except httpx.HTTPError as exc:
                st.error(
                    "Chat request failed: "
                    f"{exc}"
                )
                st.stop()

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": followup_answer,
            }
        )

        with st.chat_message(
            "assistant"
        ):
            st.write(
                followup_answer
            )