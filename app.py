
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="CarPrice PK | Price Predictor",
    page_icon="🚘",
    layout="wide",
)

# Load the existing trained pipeline
MODEL_PATH = Path(__file__).parent / "rf_car_price_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #101827, #172554);
    }
    .main .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
    }
    h1, h2, h3, p, label {
        color: #f1f5f9;
    }
    div[data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 22px;
        border-radius: 14px;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("🚘 CarPrice PK")
st.subheader("Used Car Price Prediction")
st.write(
    "Enter your car details to receive an estimated price "
    "from a machine-learning model."
)

st.divider()

# Input form
st.header("Car details")

with st.form("car_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Manufacturing year",
            min_value=1950,
            max_value=2035,
            value=2020,
            step=1,
        )

        engine_value = st.number_input(
            "Engine size / value",
            min_value=0.0,
            max_value=100000.0,
            value=1300.0,
            step=100.0,
        )

        engine_unit = st.selectbox(
            "Engine unit",
            ["cc", "kWh"],
        )

        engine_type = st.selectbox(
            "Engine type",
            ["Petrol", "Hybrid", "Diesel"],
        )

    with col2:
        transmission = st.selectbox(
            "Transmission",
            ["Automatic", "Manual"],
        )

        km_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            max_value=2000000,
            value=50000,
            step=5000,
        )

        brand = st.text_input(
            "Car brand",
            placeholder="e.g. Toyota",
        ).strip()

        model_name = st.text_input(
            "Car model",
            placeholder="e.g. Corolla",
        ).strip()

    submitted = st.form_submit_button(
        "Predict Car Price",
        type="primary",
        use_container_width=True,
    )

# Prediction
if submitted:
    if not brand or not model_name:
        st.error("Please enter both the car brand and model.")
    elif year > 2026:
        st.error("Please enter a valid manufacturing year.")
    elif engine_value <= 0:
        st.error("Engine size/value must be greater than zero.")
    else:
        car = pd.DataFrame(
            [
                {
                    "year": int(year),
                    "engine_value": float(engine_value),
                    "engine_unit": engine_unit,
                    "Engine_type": engine_type,
                    "Transmission": transmission,
                    "Km_Driven": int(km_driven),
                    "brand": brand,
                    "model_name": model_name,
                }
            ],
            columns=model.feature_names_in_,
        )

        try:
            prediction = float(model.predict(car)[0])

            if not pd.notna(prediction):
                raise ValueError("The model returned an invalid prediction.")

            if prediction < 0:
                st.warning(
                    "The model returned a negative price estimate. "
                    "This may indicate a model limitation."
                )
            else:
                st.success("Prediction completed.")

                st.metric(
                    "Estimated Car Price",
                    f"PKR {prediction:,.2f} lakh",
                )

                st.caption(
                    "This is a model-generated estimate, not a verified "
                    "market valuation or guaranteed selling price."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()

st.caption(
    "CarPrice PK | Random Forest Regression | "
    "Estimates depend on the training data and input quality."
)
