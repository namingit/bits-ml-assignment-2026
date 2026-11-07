# Breast Cancer Diagnosis – Comparison of 5 ML Classification Models

## a. Problem Statement

Breast cancer is one of the most common cancers worldwide, and early, accurate diagnosis strongly improves outcomes. The goal of this project is to build and compare machine-learning models that classify a breast mass as **malignant** or **benign** from features computed on a digitized image of a fine needle aspirate (FNA) of the mass.

This is a **binary classification** problem. Five classifiers (Logistic Regression, Decision Tree, k-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine) are trained on the same dataset and evaluated with six metrics: Accuracy, AUC, Precision, Recall, F1 score and Matthews Correlation Coefficient (MCC). The models are served through an interactive Streamlit web app.

## b. Dataset Description

| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (also on Kaggle) |
| Instances | 569 |
| Features | 30 numeric, real-valued features |
| Target | `diagnosis` – 0 = Malignant (212 samples, 37.3%), 1 = Benign (357 samples, 62.7%) |
| Missing values | None |

**Features:** ten characteristics are computed for each cell nucleus – radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. For each one, the dataset gives the **mean**, the **standard error** and the **"worst"** (mean of the three largest values), giving 10 × 3 = 30 features.

**Preprocessing and split:**
- Stratified 80/20 train-test split (`random_state=42`): 455 training rows, 114 test rows.
- Every model is a scikit-learn `Pipeline` of `StandardScaler` → classifier, so scaling is fitted on training data only.
- The 114-row held-out test set is saved as `test_data.csv` (30 features + `diagnosis`) and is what the Streamlit app evaluates.
- AUC uses predicted probabilities (SVM uses its decision-function score). Precision, Recall and F1 are reported for the positive class (1 = Benign).

## c. GitHub Repository Link

**Repository:** https://github.com/<your-username>/ml-classification-app

**Live Streamlit App:** https://<your-app-name>.streamlit.app

```
ml-classification-app/
│-- app.py                    # Streamlit app
│-- requirements.txt          # Dependencies
│-- README.md                 # This file
│-- test_data.csv             # Held-out test data (114 rows) used in the experiments
│-- model/
    │-- train_models.py       # Training + evaluation script for all 5 models
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- svm.pkl
    │-- metrics.csv           # Metrics of all models on the test set
```

**Run locally**
```bash
pip install -r requirements.txt
python model/train_models.py   # (optional) retrain models and regenerate test_data.csv
streamlit run app.py
```

## d. Models Used

| Model | Configuration |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=5000)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=5)` |
| kNN | `KNeighborsClassifier(n_neighbors=7)` |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | `SVC(kernel="rbf")` |

All models use `StandardScaler` and `random_state=42` where applicable.

### Comparison Table (evaluated on the 114-row test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Joint best on every threshold metric (Accuracy 0.9825, F1 0.9861, MCC 0.9623) and the highest AUC (0.9954). Only 2 of 114 test samples were misclassified. After scaling, the two classes are close to linearly separable in this 30-dimensional space, so a simple linear model works extremely well and is also fast and interpretable. |
| Decision Tree | Weakest model overall (Accuracy 0.9211, MCC 0.8341, AUC 0.9163). A single tree makes axis-aligned, step-wise splits and its probability outputs are coarse, which explains the lowest AUC. Even with `max_depth=5` it is a high-variance learner; an ensemble such as Random Forest would likely close the gap. |
| kNN | Strong performer (Accuracy 0.9737, MCC 0.9442) with perfect Recall (1.000): no benign case was predicted as malignant, but it produced 3 false "benign" predictions on malignant cases, which is the more costly error in a medical setting. It depends heavily on feature scaling, which the pipeline provides. |
| Naive Bayes | Moderate threshold metrics (Accuracy 0.9298, MCC 0.8492) but a high AUC (0.9868), so it ranks samples well even though its decision threshold is less accurate. Its feature-independence assumption is violated here, since radius, perimeter and area are highly correlated, which double-counts evidence and makes its probabilities over-confident. |
| Support Vector Machine | Ties Logistic Regression on Accuracy, Precision, Recall, F1 and MCC, with an almost identical AUC (0.9950). The RBF kernel with scaled features finds a near-optimal margin, but it gives no gain over the linear model, which again suggests the decision boundary is close to linear. |

### Overall Winner

**Logistic Regression** is the winner for this dataset. It ties with the SVM on Accuracy, Precision, Recall, F1 and MCC (0.9623) and has the highest AUC (0.9954), while being simpler, faster and more interpretable than the RBF-kernel SVM. The Support Vector Machine is a very close second, followed by kNN; the Decision Tree performed worst.

## Streamlit App Features

- **CSV upload** of test data (a bundled `test_data.csv` can be used or downloaded from the sidebar).
- **Model selection dropdown** for all 5 models.
- **Evaluation metrics** display: Accuracy, AUC, Precision, Recall, F1, MCC.
- **Confusion matrix** heatmap and **classification report**.
- **Comparison table and bar chart** of all 5 models on the uploaded data, plus row-level predictions.
