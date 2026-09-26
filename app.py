
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(page_title="Climate Energy Consumption Predictor", layout="wide")

st.title("⚡ Climate Energy Consumption Prediction")
st.caption(
    "A feedforward neural network that predicts energy consumption "
    "from climate and time-based features."
)

# ---------------------------------------------------------------------------
# Sidebar: data source + training settings
# ---------------------------------------------------------------------------
st.sidebar.header("1. Data")
uploaded_file = st.sidebar.file_uploader("Upload climate_energy.csv", type=["csv"])
use_sample = st.sidebar.checkbox("Use a generated sample dataset instead", value=not bool(uploaded_file))

st.sidebar.header("2. Training Settings")
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=300, value=100, step=10)
batch_size = st.sidebar.selectbox("Batch size", [8, 16, 32, 64], index=1)
test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5) / 100
val_split = st.sidebar.slider("Validation split (%)", min_value=10, max_value=40, value=20, step=5) / 100
random_state = st.sidebar.number_input("Random state", value=42, step=1)

train_button = st.sidebar.button("🚀 Train Model", type="primary")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sample_data(n=500, seed=42):
    """Synthetic fallback dataset with the same schema as climate_energy.csv."""
    rng = np.random.default_rng(seed)
    locations = ["Chennai", "Coimbatore", "Madurai", "Salem"]
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    loc = rng.choice(locations, size=n)
    temp = rng.normal(30, 5, size=n).round(1)
    base = 50 + (temp - 25) * 2.5
    hour_effect = np.sin(timestamps.hour / 24 * 2 * np.pi) * 10
    noise = rng.normal(0, 5, size=n)
    energy = base + hour_effect + noise

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "location": loc,
            "temperature": temp,
            "energy_consumption": energy.round(2),
        }
    )


@st.cache_data(show_spinner=False)
def engineer_features(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df.drop("timestamp", axis=1, inplace=True)
    return df


def build_model(input_dim: int):
    model = Sequential()
    model.add(Dense(16, activation="relu", input_shape=(input_dim,)))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if uploaded_file is not None and not use_sample:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded uploaded CSV.")
elif use_sample:
    raw_df = load_sample_data()
    st.sidebar.info("Using generated sample dataset (no file uploaded).")
else:
    st.info("Upload a CSV or check 'Use a generated sample dataset' to get started.")
    st.stop()

st.subheader("Step 1 – Raw Data Preview")
st.dataframe(raw_df.head(), use_container_width=True)

with st.expander("Dataset info & summary statistics"):
    buf_col1, buf_col2 = st.columns(2)
    with buf_col1:
        st.write("**Shape:**", raw_df.shape)
        st.write("**Dtypes:**")
        st.write(raw_df.dtypes)
    with buf_col2:
        st.write("**Describe:**")
        st.dataframe(raw_df.describe(), use_container_width=True)

required_cols = {"timestamp", "location", "energy_consumption"}
missing = required_cols - set(raw_df.columns)
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
st.subheader("Step 2 – Feature Engineering")
df = engineer_features(raw_df)
st.dataframe(df.head(), use_container_width=True)

encoder = LabelEncoder()
df["location"] = encoder.fit_transform(df["location"])
location_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
st.write("**Location encoding:**", location_map)

X = df.drop("energy_consumption", axis=1)
y = df["energy_consumption"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=int(random_state)
)

st.write(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# ---------------------------------------------------------------------------
# Train model (only when button pressed, cached in session state)
# ---------------------------------------------------------------------------
if train_button:
    with st.spinner("Training the neural network..."):
        model = build_model(X_train.shape[1])

        summary_lines = []
        model.summary(print_fn=lambda line: summary_lines.append(line))

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            verbose=0,
        )

        loss, mae_metric = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test, verbose=0)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

    st.session_state["model"] = model
    st.session_state["history"] = history.history
    st.session_state["summary_lines"] = summary_lines
    st.session_state["metrics"] = {
        "loss": loss,
        "test_mae": mae_metric,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    st.session_state["predictions"] = predictions
    st.session_state["y_test"] = y_test
    st.session_state["scaler"] = scaler
    st.session_state["encoder"] = encoder
    st.session_state["feature_columns"] = list(X.columns)
    st.success("Model trained successfully!")

# ---------------------------------------------------------------------------
# Show results if a model has been trained this session
# ---------------------------------------------------------------------------
if "model" in st.session_state:
    st.subheader("Step 3 – Model Summary")
    st.code("\n".join(st.session_state["summary_lines"]))

    st.subheader("Step 4 – Evaluation Metrics")
    m = st.session_state["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test Loss (MSE)", f"{m['loss']:.3f}")
    c2.metric("Test MAE (Keras)", f"{m['test_mae']:.3f}")
    c3.metric("RMSE", f"{m['rmse']:.3f}")
    c4.metric("MAE", f"{m['mae']:.3f}")
    c5.metric("R² Score", f"{m['r2']:.3f}")

    st.subheader("Step 5 – Training vs Validation Loss")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(st.session_state["history"]["loss"], label="Training Loss")
    ax1.plot(st.session_state["history"]["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("Step 6 – Actual vs Predicted")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(st.session_state["y_test"], st.session_state["predictions"])
    ax2.set_xlabel("Actual Energy Consumption")
    ax2.set_ylabel("Predicted Energy Consumption")
    ax2.set_title("Actual vs Predicted")
    st.pyplot(fig2)

    st.subheader("Step 7 – Prediction Error Distribution")
    errors = st.session_state["y_test"].values - st.session_state["predictions"].flatten()
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.hist(errors, bins=20)
    ax3.set_xlabel("Prediction Error")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Prediction Error Distribution")
    st.pyplot(fig3)

    # -----------------------------------------------------------------------
    # Predict new data
    # -----------------------------------------------------------------------
    st.subheader("Step 8 – Predict New Climate Data")
    encoder_trained = st.session_state["encoder"]
    feature_columns = st.session_state["feature_columns"]

    with st.form("prediction_form"):
        cols = st.columns(len(feature_columns))
        input_values = {}
        for i, col_name in enumerate(feature_columns):
            with cols[i]:
                if col_name == "location":
                    choice = st.selectbox("Location", list(encoder_trained.classes_))
                    input_values["location"] = encoder_trained.transform([choice])[0]
                elif col_name == "hour":
                    input_values["hour"] = st.number_input("Hour", 0, 23, 14)
                elif col_name == "day":
                    input_values["day"] = st.number_input("Day", 1, 31, 5)
                elif col_name == "month":
                    input_values["month"] = st.number_input("Month", 1, 12, 1)
                else:
                    input_values[col_name] = st.number_input(col_name.title(), value=0.0)

        submitted = st.form_submit_button("Predict Energy Consumption")

    if submitted:
        new_data = pd.DataFrame({col: [input_values[col]] for col in feature_columns})
        new_scaled = st.session_state["scaler"].transform(new_data)
        prediction = st.session_state["model"].predict(new_scaled, verbose=0)
        st.success(f"**Predicted Energy Consumption:** {prediction[0][0]:.2f}")
else:
    st.info("Configure settings in the sidebar and click **Train Model** to begin.")