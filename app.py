"""
Streamlit app: Breast Cancer classification - compare 5 ML models on uploaded test data.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
sys.path.insert(0, MODEL_DIR)

from train_models import (MODEL_FILES, TARGET, build_models,  # noqa: E402
                          evaluate, get_scores, load_data)

CLASS_NAMES = ["Malignant (0)", "Benign (1)"]

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")


@st.cache_resource
def load_models():
    """Load saved models; if a pickle can't be loaded (e.g. version mismatch), retrain it."""
    models, retrained = {}, []
    fresh = None
    for name, fname in MODEL_FILES.items():
        try:
            models[name] = joblib.load(os.path.join(MODEL_DIR, fname))
        except Exception:
            if fresh is None:
                from sklearn.model_selection import train_test_split
                df = load_data()
                X, y = df.drop(columns=[TARGET]), df[TARGET]
                X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                fresh = {n: m.fit(X_tr, y_tr) for n, m in build_models().items()}
            models[name] = fresh[name]
            retrained.append(name)
    return models, retrained


models, retrained = load_models()
feature_names = list(load_data().drop(columns=[TARGET]).columns)

# ---------------- Sidebar ----------------
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader("CSV file (30 features + 'diagnosis' column)", type=["csv"])
st.sidebar.caption("No file? Tick below to use the bundled test_data.csv.")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=uploaded is None)

with open(os.path.join(ROOT, "test_data.csv"), "rb") as f:
    st.sidebar.download_button("Download test_data.csv", f, file_name="test_data.csv", mime="text/csv")

st.sidebar.header("2. Choose a model")
model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

# ---------------- Main ----------------
st.title("🩺 Breast Cancer Diagnosis - ML Model Comparison")
st.write(
    "Five classifiers trained on the **UCI Breast Cancer Wisconsin (Diagnostic)** dataset "
    "(569 instances, 30 features). Upload test data to evaluate them."
)
if retrained:
    st.info(f"Saved model files were incompatible with this environment, so these were retrained: {', '.join(retrained)}")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv(os.path.join(ROOT, "test_data.csv"))
else:
    st.warning("Upload a CSV or tick 'Use bundled test_data.csv' in the sidebar.")
    st.stop()

missing = [c for c in feature_names if c not in df.columns]
if missing:
    st.error(f"The CSV is missing {len(missing)} required feature columns, e.g. {missing[:3]}")
    st.stop()

X = df[feature_names]
has_target = TARGET in df.columns

with st.expander(f"Preview uploaded data ({df.shape[0]} rows × {df.shape[1]} columns)"):
    st.dataframe(df.head(20), width="stretch")

model = models[model_name]
y_pred = model.predict(X)

if not has_target:
    st.warning(f"No '{TARGET}' column found, so only predictions are shown (metrics need true labels).")
    out = df.copy()
    out["prediction"] = pd.Series(y_pred).map({0: "Malignant", 1: "Benign"})
    st.dataframe(out, width="stretch")
    st.stop()

y = df[TARGET]

# ---- Metrics for selected model ----
st.header(f"Evaluation metrics: {model_name}")
m = evaluate(model, X, y)
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

# ---- Confusion matrix + classification report ----
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")

# ---- All-model comparison ----
st.header("Comparison of all 5 models on this data")
rows = [{"ML Model Name": n, **evaluate(mdl, X, y)} for n, mdl in models.items()]
comp = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(comp.style.highlight_max(axis=0, color="#c6efce"), width="stretch")

fig2, ax2 = plt.subplots(figsize=(10, 4))
comp.plot(kind="bar", ax=ax2, rot=0, width=0.8)
ax2.set_ylim(max(0, comp.values.min() - 0.05), 1.0)
ax2.set_ylabel("Score")
ax2.legend(loc="lower right", ncol=6, fontsize=8)
st.pyplot(fig2)

best = comp["MCC"].idxmax()
st.success(f"Best model on this data by MCC: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f})")

# ---- Predictions ----
with st.expander("Show row-level predictions"):
    out = df[[TARGET]].copy()
    out["prediction"] = y_pred
    out["score"] = get_scores(model, X)
    out["correct"] = out[TARGET] == out["prediction"]
    st.dataframe(out, width="stretch")
