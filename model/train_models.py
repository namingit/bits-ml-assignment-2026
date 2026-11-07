"""
Train and evaluate 5 classification models on the
Breast Cancer Wisconsin (Diagnostic) dataset (UCI).

Run from the project root:
    python model/train_models.py

Outputs:
    model/*.pkl           - trained model pipelines (one per model)
    model/metrics.csv     - evaluation metrics on the held-out test set
    test_data.csv         - held-out test set (features + target) for the Streamlit app
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "diagnosis"  # 1 = benign, 0 = malignant (UCI / sklearn encoding)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}


def build_models():
    """Every model is a pipeline: StandardScaler -> classifier."""
    def pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "Logistic Regression": pipe(LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        "Decision Tree": pipe(DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        "kNN": pipe(KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": pipe(GaussianNB()),
        "Support Vector Machine": pipe(SVC(kernel="rbf", random_state=RANDOM_STATE)),
    }


def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": TARGET})
    return df


def get_scores(model, X):
    """Continuous scores for AUC: probabilities if available, else SVM decision function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = get_scores(model, X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Save held-out test data for the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(HERE, MODEL_FILES[name]))
        m = evaluate(model, X_test, y_test)
        rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in m.items()}})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(HERE, "metrics.csv"), index=False)

    print(f"Dataset: {df.shape[0]} instances, {X.shape[1]} features")
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
