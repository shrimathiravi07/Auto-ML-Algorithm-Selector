"""
ML Algorithm Trainer (with UI)
==============================
Upload any CSV dataset, pick the algorithms you want to try, and this
app trains + tests every one of them and shows you the accuracy /
performance in a web UI, with a recommendation for the best model.

Supports:

Classification
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - KNN

Regression
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor

How to run
-----------
    pip install streamlit pandas scikit-learn
    streamlit run app.py

Then a browser tab opens automatically (usually http://localhost:8501).
Upload your CSV, choose the target column, pick your algorithms, and
click "Train Models".
"""

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

st.set_page_config(page_title="ML Algorithm Trainer", layout="wide")

# ----------------------------------------------------------------------
# Model registries
# ----------------------------------------------------------------------

CLASSIFICATION_MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), True),   # (model, needs_scaling)
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "Random Forest": (RandomForestClassifier(random_state=42), False),
    "KNN": (KNeighborsClassifier(), True),
}

REGRESSION_MODELS = {
    "Linear Regression": (LinearRegression(), True),
    "Decision Tree Regressor": (DecisionTreeRegressor(random_state=42), False),
    "Random Forest Regressor": (RandomForestRegressor(random_state=42), False),
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def preprocess(df: pd.DataFrame, target_col: str):
    """Encode categorical columns, fill missing values, split X/y."""
    X = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].copy()

    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.mean(numeric_only=True))

    return X, y_raw


def is_classification_target(y: pd.Series) -> bool:
    """Heuristic: text/category target, or numeric with few unique values -> classification."""
    if y.dtype == object or str(y.dtype) == "category":
        return True
    unique_ratio = y.nunique() / len(y)
    return y.nunique() <= 20 and unique_ratio < 0.05


def train_classification(X, y, selected_models, test_size):
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=42,
        stratify=y_enc if len(set(y_enc)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    for name in selected_models:
        model, needs_scaling = CLASSIFICATION_MODELS[name]
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, preds, average="weighted", zero_division=0),
        })

    return pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def train_regression(X, y, selected_models, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    for name in selected_models:
        model, needs_scaling = REGRESSION_MODELS[name]
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results.append({
            "Model": name,
            "R2 Score": r2_score(y_test, preds),
            "RMSE": rmse,
            "MAE": mean_absolute_error(y_test, preds),
        })

    return pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("🧠 ML Algorithm Trainer")
st.caption("Upload a dataset, pick your algorithms, and compare their performance.")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox("Select the target column (what you want to predict)", df.columns, index=len(df.columns) - 1)

    y_preview = df[target_col]
    default_task = "Classification" if is_classification_target(y_preview) else "Regression"

    with col2:
        task_type = st.radio("Task type", ["Classification", "Regression"],
                              index=0 if default_task == "Classification" else 1, horizontal=True)

    st.caption(f"💡 Based on your target column, this looks like a **{default_task}** problem — adjust above if needed.")

    model_options = list(CLASSIFICATION_MODELS.keys()) if task_type == "Classification" else list(REGRESSION_MODELS.keys())
    selected_models = st.multiselect("Choose algorithms to train", model_options, default=model_options)

    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5) / 100

    if st.button("🚀 Train Models", type="primary"):
        if not selected_models:
            st.warning("Please select at least one algorithm.")
        else:
            with st.spinner("Training models..."):
                X, y = preprocess(df, target_col)

                if task_type == "Classification":
                    results_df = train_classification(X, y, selected_models, test_size)
                    metric_col = "Accuracy"
                else:
                    results_df = train_regression(X, y, selected_models, test_size)
                    metric_col = "R2 Score"

            st.subheader("📊 Results")
            st.dataframe(
                results_df.style.format({c: "{:.4f}" for c in results_df.columns if c != "Model"})
                .background_gradient(subset=[metric_col], cmap="Greens"),
                use_container_width=True,
            )

            st.bar_chart(results_df.set_index("Model")[metric_col])

            best_model = results_df.iloc[0]
            st.success(
                f"🏆 **Recommended model: {best_model['Model']}** "
                f"({metric_col}: {best_model[metric_col]:.4f})"
            )
else:
    st.info("👆 Upload a CSV file to get started.")
