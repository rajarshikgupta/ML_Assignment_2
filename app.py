import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="Handwritten Digit Classification", page_icon="🔢", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

FEATURES = [f"pixel_{i}" for i in range(1, 65)]

st.title("🔢 Handwritten Digit Classification")
st.caption("Interactive comparison of five classification models")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)
    missing = [c for c in FEATURES if c not in data.columns]

    if missing or "target" not in data.columns:
        st.error("Invalid file. The CSV must contain pixel_1 through pixel_64 and target.")
        st.stop()

    X = data[FEATURES]
    y = data["target"].astype(int)

    model_name = st.selectbox("Select classification model", list(MODEL_FILES.keys()))
    model = joblib.load(MODEL_FILES[model_name])

    pred = model.predict(X)
    prob = model.predict_proba(X)

    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }

    st.subheader(f"{model_name} - Evaluation Metrics")
    cols = st.columns(6)
    for col, (metric, value) in zip(cols, metrics.items()):
        col.metric(metric, f"{value:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.code(classification_report(
        y, pred, labels=list(range(10)),
        target_names=[str(i) for i in range(10)],
        zero_division=0
    ))

    st.subheader("Prediction Preview")
    preview = X.copy()
    preview["Actual"] = y.to_numpy()
    preview["Predicted"] = pred
    st.dataframe(preview.head(20), use_container_width=True)
else:
    st.warning("Upload test_data.csv to run the evaluation.")
