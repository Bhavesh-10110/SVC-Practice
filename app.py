import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "gender_classification_v7.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

st.set_page_config(page_title="SVC Gender Predictor", layout="centered")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


st.title("SVC Gender Prediction")
st.caption("Simple SVC app based on the SVM notebook.")

if not DATA_PATH.exists():
    st.error("gender_classification_v7.csv not found in this folder.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("model.pkl not found in this folder.")
    st.stop()

# Load data and preprocess like the notebook
_df = load_data(DATA_PATH)
_df["gender"] = _df["gender"].map({"Male": 0, "Female": 1})

X = _df.iloc[:, :-1]

model = load_model()

scaler = StandardScaler()
scaler.fit(X)

st.subheader("Input Features")

user_inputs = []
for column in X.columns:
    min_value = float(X[column].min())
    max_value = float(X[column].max())
    median_value = float(X[column].median())

    value = st.number_input(
        label=column,
        min_value=min_value,
        max_value=max_value,
        value=median_value,
    )
    user_inputs.append(value)

if st.button("Predict"):
    input_df = pd.DataFrame([user_inputs], columns=X.columns)

    if hasattr(model, "n_features_in_") and model.n_features_in_ != input_df.shape[1]:
        st.error(
            f"Feature mismatch: model expects {model.n_features_in_}, "
            f"but got {input_df.shape[1]}."
        )
        st.stop()

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    label = "Female" if prediction == 1 else "Male"
    st.success(f"Prediction: {label}")
