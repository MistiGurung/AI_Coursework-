import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="EMS Transport Prediction")
st.title("EMS Transport Prediction")

st.markdown("""
**Model:** Random Forest  

**Purpose:** Predicts whether emergency transportation is **required or not required** based on priority and time-related inputs.

**Usage:** Enter incident details and click **Predict**.
""")

@st.cache_data
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    df["istransport"] = df["istransport"].astype(str).str.strip().str.lower().map({
        "true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0
    })
    df = df.dropna(subset=["istransport"])
    df["istransport"] = df["istransport"].astype(int)

    # Datetime features
    df["call_datetime"] = pd.to_datetime(df["call_datetime"], errors="coerce")
    df = df.dropna(subset=["call_datetime"])
    df["hour"] = df["call_datetime"].dt.hour
    df["day_of_week"] = df["call_datetime"].dt.dayofweek

    # Priority numeric
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce")
    df = df.dropna(subset=["priority"])
    df["priority"] = df["priority"].astype(int)

    return df

@st.cache_resource
def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    X = df[["priority", "hour", "day_of_week"]]
    y = df["istransport"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

CSV_PATH = "dataset/ems_calls.csv"  

try:
    df = load_and_prepare_data(CSV_PATH)
    model = train_model(df)
except Exception as e:
    st.error(f"Could not load data or train model. Check your CSV path.\n\nDetails: {e}")
    st.stop()


st.subheader("Make a prediction")

priority = st.number_input("Priority level (1–9)", min_value=1, max_value=9, value=3, step=1)
hour = st.number_input("Hour of incident (0–23)", min_value=0, max_value=23, value=12, step=1)

day_name = st.selectbox(
    "Day of week",
    options=[
        (0, "Monday"),
        (1, "Tuesday"),
        (2, "Wednesday"),
        (3, "Thursday"),
        (4, "Friday"),
        (5, "Saturday"),
        (6, "Sunday"),
    ],
    format_func=lambda x: x[1],
)
day = day_name[0]

if st.button("Predict"):
    user_input = np.array([[priority, hour, day]])
    pred = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0][1]

    if pred == 1:
        st.success("Transport REQUIRED")
        st.success("Advice: Ambulance transportation is recommended based on the incident details.")
        st.success(f"Confidence: {proba:.2f}")
    else:
        st.error("Transport NOT required")
        st.error("Advice: Transportation is not required at this stage; on-site assessment is advised.")
        st.error(f"Confidence: {proba:.2f}")


